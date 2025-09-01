import argparse
import asyncio
from datetime import datetime
import json
import time
from typing import TypedDict

from agents import FunctionTool
from dotenv import load_dotenv

from src.tools.file_io import read_file
from src.tools.media_understanding import (
    visual_question_answer,
    generate_transcript,
    transcribe_youtube_video,
    query_youtube_video,
)
from src.tools.search import web_search
from src.agent import LMStudioAgent, OllamaAgent, ChatCompletionsAgent


load_dotenv()

RESOURCE_BASE_DIR = "/Users/johnlingi/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/897f2dfbb5c952b5c3c1509e648381f9c7b70316/2023/validation"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GAIA agent")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument(
        "--client",
        type=str,
        default="openai",
        help="Client name",
        choices=["openai", "lmstudio", "ollama"],
    )
    parser.add_argument("--target_ids", type=str, nargs="+", help="List of target question IDs")
    return parser.parse_args()


class Question(TypedDict):
    task_id: str
    question: str
    Level: str
    file_name: str


def load_questions(file_path: str) -> list[Question]:
    with open(file_path, "r") as f:
        return json.load(f)


def create_agent(
    model_name: str, client_name: str, tools: list[FunctionTool]
) -> ChatCompletionsAgent | LMStudioAgent | OllamaAgent:
    if client_name == "openai":
        return ChatCompletionsAgent(model=model_name, tools=tools)
    elif client_name == "lmstudio":
        return LMStudioAgent(model=model_name, tools=tools)
    elif client_name == "ollama":
        return OllamaAgent(model=model_name, tools=tools)
    else:
        raise ValueError(f"Unknown client name: {client_name}")


def construct_query(question: str, extra_file_names: list[str]) -> str:
    if not extra_file_names:
        return f"{question}"
    extra = ",".join(
        [f"<context_file>{RESOURCE_BASE_DIR + '/' + fn}</context_file>" for fn in extra_file_names]
    )
    return f"{question} {extra}"


async def main(model_name: str, client_name: str, target_ids: list[str]):
    agent = create_agent(
        model_name,
        client_name,
        tools=[
            read_file,
            visual_question_answer,
            generate_transcript,
            transcribe_youtube_video,
            query_youtube_video,
            web_search,
        ],
    )
    responses = {
        "responses": [],
        "agent_settings": {
            "model": model_name,
            "client": client_name,
            "tools": [tool.name for tool in agent.tools],
        },
    }
    # Run the agent
    questions = load_questions("./questions.json")
    for q in questions:
        task_id, question, _, support_file_name = (
            q["task_id"],
            q["question"],
            q["Level"],
            q["file_name"],
        )
        if target_ids and task_id not in target_ids:
            responses["responses"].append({"task_id": task_id, "answer": "", "reason": "skipped"})
            continue

        print(f"{task_id} | QUESTION {question}")
        query = construct_query(question, [support_file_name])
        try:
            response = await agent.run(query)
            responses["responses"].append(
                {"task_id": task_id, "answer": response, "reason": "completed"}
            )
            print(f"{task_id} | QUESTION {question} | RESULT {response}")
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            responses["responses"].append({"task_id": task_id, "answer": "", "reason": "error"})

        # To avoid rate limiting. Google 2.5 Flash has 10 Requests Per Minute Max
        print("Sleeping for 6 seconds to avoid rate limiting...")
        time.sleep(6)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./test_runs/responses_{now}.json", "w") as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.model, args.client, args.target_ids))
