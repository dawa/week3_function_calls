from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_showtimes, get_now_playing_movies, get_reviews, get_random_movie, buy_ticket, confirm_ticket_purchase
import json

load_dotenv()

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI

client = AsyncOpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_now_playing_movies",
            "description": "Get movies that are in theaters now. Call this whenever you need to know what's playing now.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_showtimes",
            "description": "Get movie showtimes at specific locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The movie's name or title.",
                    },
                    "location": {
                        "type": "string",
                        "description": "The location as a city, state or zipcode.",
                    },
                },
                "required": ["title", "location"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_movie",
            "description": "Select a random movie from a list of movie titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movies": {
                        "type": "string",
                        "description": "A list of movie titles.",
                    },
                },
                "required": ["movies"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": "Get reviews for a specific movie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "string",
                        "description": "The movie ID.",
                    },
                },
                "required": ["movie_id"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buy_ticket",
            "description": "To assist with ticket purchases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "theater": {
                        "type": "string",
                        "description": "The movie theater.",
                    },
                    "movie_id": {
                        "type": "string",
                        "description": "The movie ID.",
                    },
                    "showtime": {
                        "type": "string",
                        "description": "The time the movie is showing at the selected theater.",
                    },
                },
                "required": ["theater", "movie_id", "showtime"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_ticket_purchase",
            "description": "To confirm with user before ticket purchases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "theater": {
                        "type": "string",
                        "description": "The movie theater.",
                    },
                    "movie_id": {
                        "type": "string",
                        "description": "The movie ID.",
                    },
                    "showtime": {
                        "type": "string",
                        "description": "The time the movie is showing at the selected theater.",
                    },
                },
                "required": ["theater", "movie_id", "showtime"],
                "additionalProperties": False,
            },
        }
    }
]

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "tools": tools,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful movie chatbot that helps people explore movies in theaters. \
Use the supplied tools to assist the user. Be clear and concise. Ask for clarification if needed. Keep a friendly and helpful tone.
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    # Handle streamed response
    function_call_data = None
    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        token_content = part.choices[0].delta.content or ""
        await response_message.stream_token(token_content)

        # Check for a function call
        if part.choices[0].delta.get("function_call"):
            function_call_data = part.choices[0].delta.function_call
    
    await response_message.update()
    
    return response_message, function_call_data

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    # Generate response and extract any potential function call data
    response_message, function_call_data = await generate_response(client, message_history, gen_kwargs)

    if function_call_data:
        try:
            json_message = json.loads(function_call_data['arguments'])
            function_name = function_call_data['name']

            # Call the appropriate function based on the function name
            if function_name == "get_showtimes":
                title = json_message.get("title")
                location = json_message.get("location")
                result = get_showtimes(title, location)
            elif function_name == "get_now_playing_movies":
                result = get_now_playing_movies()
            elif function_name == "get_random_movie":
                movie_list = json_message.get("movies")
                result = get_random_movie(movie_list)
            elif function_name == "get_reviews":
                movie_id = json_message.get("movie_id")
                result = get_reviews(movie_id)
            elif function_name == "buy_ticket":
                movie_id = json_message.get("movie_id")
                theater = json_message.get("theater")
                showtime = json_message.get("showtime")
                result = buy_ticket(theater, movie_id, showtime)
            elif function_name == "confirm_ticket_purchase":
                movie_id = json_message.get("movie_id")
                theater = json_message.get("theater")
                showtime = json_message.get("showtime")
                result = confirm_ticket_purchase(theater, movie_id, showtime)
            else:
                result = "Unknown function call: " + function_name

            # Append the result to the message history and send it to the user
            message_history.append({"role": "system", "content": result})
            cl.user_session.set("message_history", message_history)

            response_message = await generate_response(client, message_history, gen_kwargs)

        except json.JSONDecodeError:
            print(f"Error: Unable to parse the message as JSON {response_message.content}")

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
