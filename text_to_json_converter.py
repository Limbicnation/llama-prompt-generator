import json
import os
import random

# Function to parse the text and convert it to a structured JSON format
def parse_text_to_json(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    current_entry = {}
    used_seeds = set()

    for line in lines:
        line = line.strip()
        if line.startswith("Prompt:"):
            if current_entry:
                # Ensure unique seed
                while True:
                    seed = random.randint(0, 999999)
                    if seed not in used_seeds:
                        used_seeds.add(seed)
                        current_entry['Seed'] = seed
                        break
                data.append(current_entry)
                current_entry = {}
            current_entry['Prompt'] = line.replace("Prompt:", "").strip()
        elif line.startswith("Generated:"):
            current_entry['Generated'] = line.replace("Generated:", "").strip()
        elif line.startswith("Model:"):
            current_entry['Model'] = line.replace("Model:", "").strip()
        elif line.startswith("Date:"):
            current_entry['Date'] = line.replace("Date:", "").strip()

    if current_entry:
        # Ensure unique seed for the last entry
        while True:
            seed = random.randint(0, 999999)
            if seed not in used_seeds:
                used_seeds.add(seed)
                current_entry['Seed'] = seed
                break
        data.append(current_entry)

    return data

# File path to the input text file
file_path = '/mnt/TurboTux/AnacondaWorkspace/Github/meta-llama-demo/generated_texts_Layers/meta_llama_generated_prompts_20240703_v33.txt'

# Parse the text to JSON
parsed_data = parse_text_to_json(file_path)

# Convert to JSON string
json_data = json.dumps(parsed_data, indent=4)

# Output path for the JSON file within your current workspace
output_path = '/mnt/TurboTux/AnacondaWorkspace/Github/meta-llama-demo/meta_llama_generated_prompts_20240703_v33.json'

# Ensure the output directory exists
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# Save the JSON data to a file
with open(output_path, 'w') as json_file:
    json_file.write(json_data)

print(f"JSON file has been saved to: {output_path}")
