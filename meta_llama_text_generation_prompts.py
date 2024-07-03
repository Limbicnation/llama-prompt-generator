import json

with open("tokenized_examples.json", "r") as f:
    prompts = json.load(f)
    num_elements = len(prompts)

print(f"Number of elements: {num_elements}")
