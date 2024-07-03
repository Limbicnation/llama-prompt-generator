import json
import argparse
import os
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List
from datetime import datetime

def load_model(model_name: str) -> Tuple[AutoTokenizer, transformers.pipelines.TextGenerationPipeline]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return tokenizer, model_pipeline

def generate_text(pipeline: transformers.pipelines.TextGenerationPipeline, prompt: str, tokenizer: AutoTokenizer, max_length: int = 400) -> List[str]:
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        truncation=True,
        max_length=max_length,
    )
    return [seq['generated_text'] for seq in sequences]

def generate_output_suffix() -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    suffix = f"meta_llama_generated_prompts_{date_str}"
    return suffix

def load_prompts(file_path: str, themes: List[str] = None) -> List[str]:
    with open(file_path, "r") as f:
        prompts = json.load(f)
    
    if themes:
        themes_set = set(themes)
        valid_prompts = [prompt['text'] for prompt in prompts if themes_set.intersection(prompt['themes'])]
    else:
        valid_prompts = [prompt['text'] for prompt in prompts]
    
    return valid_prompts

def generate_prompts_from_model(pipeline: transformers.pipelines.TextGenerationPipeline, themes: List[str], tokenizer: AutoTokenizer, max_length: int = 50) -> List[str]:
    prompts = []
    for theme in themes:
        prompt = f"Generate a prompt about {theme}"
        sequences = generate_text(pipeline, prompt, tokenizer, max_length)
        prompts.extend(sequences)
    return prompts

def get_versioned_filename(base_filename: str) -> str:
    version = 1
    while os.path.exists(f"{base_filename}_v{version:02d}.txt"):
        version += 1
    return f"{base_filename}_v{version:02d}.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with specified themes and limit output count.")
    parser.add_argument('--themes', nargs='+', help='List of themes to filter prompts by.')
    parser.add_argument('--output_count', type=int, default=10, help='Total number of output prompts to generate.')
    parser.add_argument('--log_generation', action='store_true', help='Enable detailed logging of the generation process.')
    parser.add_argument('--include_metadata', action='store_true', help='Include additional metadata in the generated text outputs.')
    parser.add_argument('--use_model_prompts', action='store_true', help='Generate prompts directly using the model instead of loading from a file.')
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, pipeline = load_model(model_name)
    
    if args.use_model_prompts:
        valid_prompts = generate_prompts_from_model(pipeline, args.themes, tokenizer)
    else:
        valid_prompts = load_prompts("tokenized_examples.json", themes=args.themes)
    
    if not valid_prompts:
        print("No valid prompts found for the specified themes.")
        exit()

    output_suffix = generate_output_suffix()
    output_dir = os.path.dirname(f"generated_texts_{output_suffix}.txt") or '.'
    os.makedirs(output_dir, exist_ok=True)
    versioned_filename = get_versioned_filename(os.path.join(output_dir, f"{output_suffix}"))

    with open(versioned_filename, "w") as out_f:
        for i, prompt in enumerate(valid_prompts):
            if i >= args.output_count:
                break
            try:
                sequences = generate_text(pipeline, prompt, tokenizer)
                for seq in sequences:
                    if args.include_metadata:
                        metadata = f"Prompt: {prompt}\nGenerated: {seq}\nModel: {model_name}\nDate: {datetime.now()}\n"
                    else:
                        metadata = f"Prompt: {prompt}\nGenerated: {seq}\n"
                    out_f.write(metadata)
                    print(metadata)
                    if args.log_generation:
                        print(f"Logging: Prompt '{prompt}' generated successfully.")
            except Exception as e:
                print(f"Error generating text for prompt '{prompt}': {e}")
                if args.log_generation:
                    print(f"Logging: Error generating text for prompt '{prompt}': {e}")
