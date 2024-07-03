# Llama Prompt Generator

![Llama Prompt Generator](https://github.com/Limbicnation/meta-llama-demo/blob/main/images/Llama-Prompt-Generator.jpg)


This repository contains a demo script for running Meta Llama on Linux using PyTorch.


## Requirements

- Python
- PyTorch
- Transformers
- Hugging Face CLI
- bitsandbytes
- datasets
- PEFT
- numpy

## Setup

1. **Install Required Packages**

    ```sh
    pip install torch transformers huggingface_hub[cli] bitsandbytes datasets peft numpy
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

Run the ```generate_text.py``` script to generate text based on the prompts and themes. 🚀🚀


You can now use these updated commands for additional info. 


- `--themes`: Include themes by specifying common stable diffusion themes.
   - Example: `--themes scifi`
- `--output_count`: Control the number of output logs by specifying the desired number.
   - Example: `--output_count 1200`
- `--include_metadata`: Include additional metadata in the generated text outputs.
- `--log_generation`: Enable detailed logging of the generation process.
- `--use_model_prompts`: Generate prompts directly using the model instead of loading from tokenized_examples.json
  `--seed`: Set a random seed for reproducibility.
        Example: --seed 51544



```sh
python generate_text.py --themes scifi fantasy horror art --output_count 1200 --log_generation --include_metadata
```
🚀🚀

```
python generate_text.py --themes scifi fantasy horror art --output_count 1200 --log_generation --include_metadata --use_model_prompts --seed 51544 

```

# Dataset

This project uses the [MetaLlama Text Generation Prompts dataset](https://huggingface.co/datasets/Limbicnation/MetaLlama_Text_Generation_Prompts).

It's a prompt extender/enhancer based on: [Stable Diffusion Prompts](https://huggingface.co/datasets/daspartho/stable-diffusion-prompts)
