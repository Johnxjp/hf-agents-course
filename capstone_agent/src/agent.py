import os
from dotenv import load_dotenv

from agents import (
    Agent,
    FunctionTool,
    OpenAIChatCompletionsModel,
    Runner,
)
from openai import AsyncOpenAI

load_dotenv()

SYSTEM_PROMPT = """
You are a general AI assistant. I will ask you a question. 
Report your thoughts, and finish your answer with the following template: 
<final_answer>[YOUR FINAL ANSWER]</final_answer>. [YOUR FINAL ANSWER] should be replaced with a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use strings e.g. 'three' instead of '3', don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Some questions require processing context files. The file paths are contained in the <context_file> tags.
For example <context_file>/path/to/example_file.txt</context_file> contains the file /path/to/example_file.txt

First think step by step considering also the tools available.

ONLY respond using the following format. Do not include any additional explanations or context.
'<final_answer>[YOUR FINAL ANSWER]</final_answer>'
"""


def parse_response(response: str) -> str:
    """Extracts the final answer from the response <final_answer>"""
    if "<final_answer>" in response:
        result = response.split("<final_answer>")[1].split("</final_answer>")[0].strip()
        return f"FINAL ANSWER: {result}"
    return "FINAL ANSWER: Unknown"


class ChatCompletionsAgent:
    def __init__(self, model: str, tools: list[FunctionTool]):
        self.model = model
        self.tools = tools
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.agent = Agent(
            name="GAIA Agent",
            instructions=SYSTEM_PROMPT,
            model=OpenAIChatCompletionsModel(self.model, openai_client=self._client),
            tools=self.tools,
        )

    async def run(self, question: str) -> str:
        result = await Runner.run(self.agent, question)
        return parse_response(result.final_output)


class LMStudioAgent:
    def __init__(self, model: str, tools: list[FunctionTool]):
        self.model = model
        self.tools = tools
        self.agent = Agent(
            name="GAIA Agent",
            instructions=SYSTEM_PROMPT + "\nReasoning: medium",
            # model="gpt-4o-mini",
            model=OpenAIChatCompletionsModel(
                model=self.model,
                openai_client=AsyncOpenAI(base_url="http://localhost:1234/v1", api_key=""),
            ),
            tools=tools,
        )
        print("No models settings set. Using default")
        print(self.agent.model_settings)

    async def run(self, question: str) -> str:
        result = await Runner.run(self.agent, question)
        return parse_response(result.final_output)


class OllamaAgent:
    def __init__(self, model: str, tools: list[FunctionTool]):
        self.model = model
        self.tools = tools
        self.agent = Agent(
            name="GAIA Agent",
            instructions=SYSTEM_PROMPT + "\nReasoning: medium",
            tools=self.tools,
        )

    async def run(self, question: str) -> str:
        result = await Runner.run(self.agent, question)
        return parse_response(result.final_output)
