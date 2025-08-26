import asyncio
import datasets
from llama_index.core.schema import Document
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

with open("hf_token.txt") as f:
    hf_token = f.read().strip()

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token, provider="auto"
)

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        text="\n".join(
            [
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}",
            ]
        ),
        metadata={"name": guest_dataset["name"][i]},
    )
    for i in range(len(guest_dataset))
]

bm25_retriever = BM25Retriever.from_defaults(nodes=docs)


def get_guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if not results:
        return "No relevant guest information found."
    return "\n\n".join([f"{doc.text}" for doc in results[:3]])


guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)

async def main():
    try:
        response = await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")
        print("🎩 Alfred's Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Give time for cleanup
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())