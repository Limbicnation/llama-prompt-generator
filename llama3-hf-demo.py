import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model with correct configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create the pipeline
    model_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return tokenizer, model_pipeline

def generate_text(pipeline, prompt, tokenizer, max_length=400):
    # Generate text using the pipeline
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
    return sequences

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, pipeline = load_model(model_name)
    
    # Load prompts from the text file
    with open("tokenized_examples.txt", "r") as f:
        prompts = f.readlines()
    
    for prompt in prompts:
        prompt = prompt.strip()
        sequences = generate_text(pipeline, prompt, tokenizer)
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
