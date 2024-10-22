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

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, pipeline = load_model(model_name)
    
    try:
        # Load prompts from the text file
        with open("tokenized_examples.txt", "r") as f:
            prompts = f.readlines()
        
        # Validate and preprocess prompts
        valid_prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        
        if not valid_prompts:
            print("No valid prompts found in the file.")
            exit()
        
        # Open a file to save the generated prompts
        with open("generated_prompts.txt", "w") as out_f:
            # Generate and save text for each prompt
            for prompt in valid_prompts:
                try:
                    sequences = generate_text(pipeline, prompt, tokenizer)
                    for seq in sequences:
                        out_f.write(f"Prompt: {prompt}\nGenerated: {seq}\n\n")
                        print(f"Prompt: {prompt}\nGenerated: {seq}\n")
                except Exception as e:
                    print(f"Error generating text for prompt '{prompt}': {e}")
    
    except FileNotFoundError:
        print("The file 'tokenized_examples.txt' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
