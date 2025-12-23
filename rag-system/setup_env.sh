#!/bin/bash

# Create necessary directories
mkdir -p ./vector_store
mkdir -p ./model/lora_adapter
mkdir -p ./data

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft langchain langchain-community langchain-huggingface \
    langchain-chroma modelscope sentence-transformers accelerate bitsandbytes

echo "Environment setup complete. Place your .txt or .pdf files in the ./data folder."