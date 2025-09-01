#!/usr/bin/env python3
"""
Simple example of multi-provider fine-tuning using ONE unified approach.
"""

import asyncio
import os
from pathlib import Path
import json

# Add the parent directory to path to import src modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import finetune


def create_sample_data():
    """Create a sample training data file."""
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 equals 4."}
            ]
        }
    ]
    
    os.makedirs("data", exist_ok=True)
    
    with open("data/training_sample.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print("Created sample training data at data/training_sample.jsonl")


async def main():
    """Run examples for all providers using the same simple API."""
    
    create_sample_data()
    
    print("\n" + "="*50)
    print("Multi-Provider Fine-tuning Examples")
    print("="*50 + "\n")
    
    # Check which providers are configured
    has_openai = "OPENAI_API_KEY" in os.environ
    has_together = "TOGETHER_API_KEY" in os.environ
    has_vertex = "VERTEX_PROJECT_ID" in os.environ
    
    if not any([has_openai, has_together, has_vertex]):
        print("⚠️  No API keys found! Set one of: OPENAI_API_KEY, TOGETHER_API_KEY, VERTEX_PROJECT_ID")
        return
    
    # Same simple API for all providers!
    
    if has_openai:
        print("=== OpenAI ===")
        try:
            result = await finetune(
                provider="openai",
                model="gpt-4o-mini-2024-07-18", 
                training_file="data/training_sample.jsonl",
                suffix="openai-test",
                verbose=True,
                n_epochs=1
            )
            print(f"✓ OpenAI model: {result.model_name}\n")
        except Exception as e:
            print(f"✗ OpenAI failed: {e}\n")
    
    if has_together:
        print("=== Together AI ===")
        try:
            result = await finetune(
                provider="together",
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
                training_file="data/training_sample.jsonl", 
                suffix="together-test",
                verbose=True,
                n_epochs=1,
                lora=True
            )
            print(f"✓ Together model: {result.model_name}\n")
        except Exception as e:
            print(f"✗ Together failed: {e}\n")
    
    if has_vertex:
        print("=== Vertex AI ===")
        try:
            result = await finetune(
                provider="vertex",
                model="gemini-2.0-flash-001",
                training_file="gs://your-bucket/training_sample.jsonl",  # Must be in GCS
                suffix="vertex-test", 
                verbose=True,
                epochs=1
            )
            print(f"✓ Vertex model: {result.model_name}\n")
        except Exception as e:
            print(f"✗ Vertex failed: {e}\n")
    
    print("=== Unsloth (Local) ===")
    try:
        result = await finetune(
            provider="unsloth",
            model="llama-3.1-8b-unsloth-bnb-4bit",
            training_file="data/training_sample.jsonl",
            suffix="local-test",
            verbose=True,
            max_steps=5,  # Very short for demo
            load_in_4bit=True,
            use_lora=True
        )
        print(f"✓ Unsloth output: {result.model_name}")
    except Exception as e:
        print(f"✗ Unsloth failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())