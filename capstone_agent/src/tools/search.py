"""
Required tools:
- Web Search: can use Google
- File Read
- Image Understanding
- Audio Understanding
"""

from google import genai
from google.genai import types

from agents import function_tool
from dotenv import load_dotenv

load_dotenv()


client = genai.Client()
GOOGLE_GEMINI_MODEL = "gemini-2.0-flash"


@function_tool
def web_search(query: str) -> str:
    """
    Answers a question using web search.
    """
    print(f"Searching web for answers to {query}")
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model=GOOGLE_GEMINI_MODEL,
        config=config,
        contents={query},
    )
    print(response.text)
    return response.text
