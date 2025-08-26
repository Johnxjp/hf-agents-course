import asyncio
import random

from huggingface_hub import list_models
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

with open("hf_token.txt") as f:
    hf_token = f.read().strip()


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


# Initialize the tool
weather_info_tool = FunctionTool.from_defaults(get_weather_info)

# https://programmablesearchengine.google.com/controlpanel/overview?cx=723340f8b0e974862
tool_spec = GoogleSearchToolSpec(
    key="AIzaSyCZbBPsT6kN9etyrOjlN79wcyJfiPqiJQQ", engine="723340f8b0e974862"
)
search_tool = FunctionTool.from_defaults(tool_spec.google_search)


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


# Initialize the tool
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)


# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token, provider="auto"
)
# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [search_tool, weather_info_tool, hub_stats_tool], llm=llm
)


async def main():
    # Example query Alfred might receive during the gala
    try:
        response = await alfred.run("What is Facebook and what's their most popular model?")
        print("🎩 Alfred's Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Give time for cleanup
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
