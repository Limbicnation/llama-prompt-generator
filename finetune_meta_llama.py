from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import datasets
import torch
import bitsandbytes as bnb  # Import bitsandbytes for quantization
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Union, List

# Clear GPU memory
torch.cuda.empty_cache()

# Load dataset
dataset = datasets.load_dataset("daspartho/stable-diffusion-prompts", split="train")

# Print the column names to identify the correct field
print("Dataset columns:", dataset.column_names)

# Prepare model and tokenizer
model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Add a padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', load_in_8bit=True)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# Configure PEFT with LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Ensure the model's embedding layer is resized to account for the new padding token
model.resize_token_embeddings(len(tokenizer))

# Function to cast training parameters to FP32
def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # Only upcast trainable parameters into FP32
            if param.requires_grad:
                param.data = param.to(dtype)

# Cast trainable parameters to FP32
cast_training_params(model)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device, dtype=torch.float16)

# Prepare a text file to save tokenized examples
output_file = "tokenized_examples.txt"
with open(output_file, "w") as f:
    pass  # Clear the file if it exists

# Tokenize dataset with added checks and logging
def tokenize_function(examples):
    print(f"Examples to tokenize: {examples['prompt'][:2]}")  # Log the first few examples
    # Ensure inputs are in the expected format
    if isinstance(examples["prompt"], list) and all(isinstance(item, str) for item in examples["prompt"]):
        try:
            inputs = tokenizer(
                examples["prompt"], 
                padding='max_length', 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            # Save the examples to a text file
            with open(output_file, "a") as f:
                for prompt in examples["prompt"]:
                    f.write(prompt + "\n")
            inputs["labels"] = inputs["input_ids"].clone()  # Set labels for training
            return inputs
        except Exception as e:
            print(f"Error during tokenization: {e}")
            print(f"Examples: {examples['prompt']}")
            raise e
    else:
        raise ValueError("Input format is incorrect. Expected a list of strings.")

# Tokenize the dataset
try:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")
    print("Inspecting dataset entries for anomalies:")
    for i, entry in enumerate(dataset):
        print(f"Entry {i}: {entry}")
    raise e

# Data collator to dynamically pad the inputs
data_collator = DataCollatorWithPadding(tokenizer)

# Prepare data loader
train_loader = DataLoader(tokenized_datasets, batch_size=1, shuffle=True, collate_fn=data_collator)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # Reduce batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Use gradient accumulation
    num_train_epochs=3,
    logging_dir="./logs",
    dataloader_pin_memory=False,  # Avoid memory spikes
)

# Prepare optimizer and scaler
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()

# Custom training loop
model.train()
for epoch in range(training_args.num_train_epochs):
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        with autocast():
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Check for inf or NaN gradients before optimizer step
        found_inf = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    found_inf = True
                    break

        if not found_inf:
            scaler.step(optimizer)
            scaler.update()
        else:
            print(f"Found inf or NaN gradients at epoch {epoch}, step {step}. Skipping optimizer step.")

        if step % training_args.gradient_accumulation_steps == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

print("Training complete.")