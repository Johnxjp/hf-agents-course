from huggingface_hub import InferenceClient

from prompts import SYSTEM_PROMPT

with open("hf_token.txt") as f:
    hf_token = f.read().strip()

client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", token=hf_token)


def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=1024,
    stop=["Observation:"]
)
# print(output.choices[0].message.content)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
    {"role": "assistant", "content": output.choices[0].message.content + "Observation:\n" + get_weather('London')},
]
output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)
