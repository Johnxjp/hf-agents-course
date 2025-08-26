import asyncio
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context




def multiply_int(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


with open("hf_token.txt") as f:
    hf_token = f.read().strip()

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token, provider="auto"
)

agent = AgentWorkflow.from_tools_or_functions([FunctionTool.from_defaults(multiply_int)], llm=llm)
ctx = Context(agent)


def run_agent(input_text: str) -> str:
    response = agent.run(input_text)
    response = agent.run("My name is Bob.", ctx=ctx)
    response = agent.run("What was my name again?", ctx=ctx)
    return response


response = run_agent("What is 2 times 2?")
print(response)
