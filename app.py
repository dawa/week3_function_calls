from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_showtimes, get_now_playing_movies, get_reviews, get_random_movie, buy_ticket, confirm_ticket_purchase
import json

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful movie chatbot that helps people explore movies that are out in \
theaters. If a user asks for recent information, output a function call and \
the system add to the context. If you need to call a function, only output the \
function call. Call functions using Python syntax in plain text, no code blocks.

You have access to the following functions:
   - **get_now_playing_movies():** For currently showing films.
   - **get_showtimes():** For movie times at specific locations.
   - **get_reviews():** For recent reviews or audience reactions.
   - **get_random_movie():** To pick a random movie.
   - **buy_ticket():** To assist with ticket purchases.
   - **confirm_ticket_purchase():** To confirm with user before ticket purchases.

Generate function calls in the following format:
{ "function": "get_showtimes", "title": "movieTitle", "location": "city, state"}

{ "function": "get_now_playing_movies"}

{ "function": "get_random_movie": , "movies": "movie_list"}

{ "function": "get_reviews", "movie_id": "movieId"}

{ "function": "buy_ticket", "theater": "theater", "movie_id": "movieId", "showtime": "showtime"}

{ "function": "confirm_ticket_purchase", "theater": "theater", "movie_id": "movieId", "showtime": "showtime"}

Be clear and concise. Ask for clarification if needed. Keep a friendly and helpful tone. 
Always repeat the ticket details and display an instruction to confirm the ticket with the user when buying tickets
"""

REVIEW_PROMPT = """\
Based on the conversation, determine if the topic is about a specific movie. \
Determine if the user is asking a question that would be aided by knowing what \
critics are saying about the movie. Determine if the reviews for that movie have \
already been provided in the conversation. If so, do not fetch reviews.

Your only role is to evaluate the conversation, and decide whether to fetch reviews.

Output the current movie, id, a boolean to fetch reviews in JSON format, and your
rationale. Do not output as a code block.

{
    "movie": "title",
    "id": 123,
    "fetch_reviews": true
    "rationale": "reasoning"
}
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

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    message_history.append({"role": "system", "content": REVIEW_PROMPT})
    response_message = await generate_response(client, message_history, gen_kwargs)

    try:
        context_json = json.loads(response_message.content)

        if context_json.get("fetch_reviews", False):
            movie_id = context_json.get("id")
            reviews = get_reviews(movie_id)
            reviews = f"Reviews for {context_json.get('movie')} (ID: {movie_id}):\n\n{reviews}"
            context_message = {"role": "system", "content": f"CONTEXT: {reviews}"}
            message_history.append(context_message)
    except json.JSONDecodeError:
            print("Error: Unable to parse the review message as JSON " + response_message.content)
            json_message = None

    message_history.append({"role": "system", "content": SYSTEM_PROMPT})
    response_message = await generate_response(client, message_history, gen_kwargs)

    if response_message.content.find("{ \"function\": ") != -1:
        try:
            json_message = json.loads(response_message.content)
            function_name = json_message.get("function")

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
            elif function_name == "buy_tickets":
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

            message_history.append({"role": "system", "content": result})

            response_message = await generate_response(client, message_history, gen_kwargs)
        except json.JSONDecodeError:
            print("Error: Unable to parse the message as JSON " + response_message.content)
            json_message = None

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
