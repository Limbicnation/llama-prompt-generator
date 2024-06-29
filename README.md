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

<<<<<<< HEAD
```sh
pip install -U "huggingface_hub[cli]"
pip install peft transformers datasets torch bitsandbytes
huggingface-cli login
=======
1. **Install Required Packages**
>>>>>>> 653e21c (Update README with setup instructions, usage, and troubleshooting steps)

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
