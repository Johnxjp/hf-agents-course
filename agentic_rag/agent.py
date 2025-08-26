import asyncio

# Import necessary libraries
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from .tools import search_tool, weather_info_tool, hub_stats_tool
from .retriever import guest_info_tool

with open("hf_token.txt") as f:
    hf_token = f.read().strip()

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token, provider="together"
)

# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)


async def main():
    try:
        while True:
            user_input = input("Query: ")
            if user_input.lower() in {"exit", "quit"}:
                break
            response = await alfred.run(user_input)
            print("🎩 Alfred's Response:")
            print(response)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Give time for cleanup
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
