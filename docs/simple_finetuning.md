# Simple Multi-Provider Fine-tuning

One unified API for fine-tuning with any provider.

## Usage

### Single Fine-tuning (Experiment 1)

```python
from src.utils import finetune

# Same API for all providers!
result = await finetune(
    provider="together",  # or "openai", "vertex", "unsloth"
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    training_file="data/training.jsonl",
    suffix="my-model",
    verbose=True,
    # Provider-specific hyperparameters:
    n_epochs=3,
    lora=True,
    learning_rate=1e-5
)

print(f"Model: {result.model_name}")
```

### Iterative Fine-tuning (Experiment 2)

```python
from src.finetuners import UnifiedFinetuner

finetuner = UnifiedFinetuner(
    provider="together",  # or any provider
    hyperparameters={"n_epochs": 3, "lora": True}
)

# Use in expert iteration
result_files, model_name = await finetuner.run(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    input_log=training_log,
    log_dir="logs/iteration",
    suffix="iter-1"
)
```

## Setup

Set environment variables for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Together AI  
export TOGETHER_API_KEY="your-key"

# Vertex AI
export VERTEX_PROJECT_ID="your-project"
export VERTEX_LOCATION="us-central1"

# Unsloth (local, no keys needed)
pip install unsloth torch transformers datasets trl
```

## Provider Models

- **OpenAI**: `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06`
- **Together**: `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference`, etc.
- **Vertex**: `gemini-2.0-flash-001`, `gemini-2.0-pro-001`
- **Unsloth**: `llama-3.1-8b-unsloth-bnb-4bit`, etc.

## Data Format

JSONL with conversational format:

```json
{"messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]}
```

That's it! One simple API, any provider.
