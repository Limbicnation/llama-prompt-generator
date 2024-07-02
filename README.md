```markdown
# Meta Llama Demo

This repository contains a demo script for running Meta Llama on Linux using PyTorch.

## Requirements

- Python
- PyTorch
- Transformers
- Hugging Face CLI
- bitsandbytes
- datasets
- PEFT

## Setup

1. **Install Required Packages**

    ```sh
    pip install torch transformers huggingface_hub[cli] bitsandbytes datasets peft
    ```

2. **Login to Hugging Face CLI**

    ```sh
    huggingface-cli login
    ```

## Usage

### Finetune the Model

Run the `finetune_meta_llama.py` script to finetune the model on the stable-diffusion prompts dataset.

```sh
python finetune_meta_llama.py
```

### Generate Text

Run the `generate_text.py` script to generate text based on the prompts and themes.

```sh
python generate_text.py --themes scifi --output_count 10 🚀🚀
```
