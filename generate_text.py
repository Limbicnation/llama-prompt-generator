import json
import argparse
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, List

def load_model(model_name: str) -> Tuple[AutoTokenizer, transformers.pipelines.TextGenerationPipeline]:
    """
    Load the tokenizer and model with correct configurations.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Tuple[AutoTokenizer, transformers.pipelines.TextGenerationPipeline]: The loaded tokenizer and text generation pipeline.
    """
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
    """
    Generate text using the provided pipeline.

    Args:
        pipeline (transformers.pipelines.TextGenerationPipeline): The text generation pipeline.
        prompt (str): The prompt text to generate text from.
        tokenizer (AutoTokenizer): The tokenizer used by the pipeline.
        max_length (int): The maximum length of the generated text.

    Returns:
        List[str]: The generated text sequences.
    """
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  # Ensure pad_token_id is set
        truncation=True,
        max_length=max_length,
    )
    return [seq['generated_text'] for seq in sequences]

def load_prompts(file_path: str, themes: List[str] = None) -> List[str]:
    """
    Load prompts from a JSON file and filter them based on the provided themes.

    Args:
        file_path (str): The path to the JSON file containing prompts.
        themes (List[str], optional): List of themes to filter prompts by. Defaults to None.

    Returns:
        List[str]: The filtered list of prompts.
    """
    with open(file_path, "r") as f:
        prompts = json.load(f)
    
    if themes:
        themes_set = set(themes)
        valid_prompts = [prompt['text'] for prompt in prompts if themes_set.intersection(prompt['themes'])]
    else:
        valid_prompts = [prompt['text'] for prompt in prompts]
    
    return valid_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with specified themes and limit output count.")
    parser.add_argument('--themes', nargs='+', help='List of themes to filter prompts by.')
    parser.add_argument('--output_count', type=int, default=10, help='Total number of output prompts to generate.')
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, pipeline = load_model(model_name)
    
    # Load and filter prompts
    valid_prompts = load_prompts("tokenized_examples.json", themes=args.themes)
    
    if not valid_prompts:
        print("No valid prompts found for the specified themes.")
        exit()

    # Open a file to save the generated texts
    with open("generated_texts.txt", "w") as out_f:
        # Generate and save text for each prompt
        for i, prompt in enumerate(valid_prompts):
            if i >= args.output_count:
                break
            try:
                sequences = generate_text(pipeline, prompt, tokenizer)
                for seq in sequences:
                    out_f.write(f"Prompt: {prompt}\nGenerated: {seq}\n\n")
                    print(f"Prompt: {prompt}\nGenerated: {seq}\n")
            except Exception as e:
                print(f"Error generating text for prompt '{prompt}': {e}")
