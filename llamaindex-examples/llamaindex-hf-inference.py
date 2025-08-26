from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

with open("hf_token.txt") as f:
    hf_token = f.read().strip()

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)

response = llm.complete("Write a Python function to add two numbers.")
print(response)