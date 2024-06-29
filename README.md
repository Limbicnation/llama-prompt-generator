# Meta Llama Demo

This repository contains a demo script for running Meta Llama on Linux using PyTorch.

## Requirements

- Python
- PyTorch
- Transformers
- Hugging Face CLI

## Setup

```sh
pip install -U "huggingface_hub[cli]"
pip install peft transformers datasets torch bitsandbytes
huggingface-cli login

