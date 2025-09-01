# Multi-Provider Fine-tuning Guide

This guide explains how to use the unified fine-tuning system that supports OpenAI, Together AI, Vertex AI (Google Gemini), and Unsloth.

## Setup

### Environment Variables

Set the appropriate environment variables for the providers you plan to use:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Together AI
export TOGETHER_API_KEY="your-together-api-key"

# Vertex AI (Google Cloud)
export VERTEX_PROJECT_ID="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"  # or your preferred location
export VERTEX_GCS_BUCKET="your-gcs-bucket"  # for storing training data

# Optional: Weights & Biases for monitoring
export WANDB_API_KEY="your-wandb-api-key"
```

### Installation

Install provider-specific dependencies:

```bash
# Core dependencies
pip install openai asyncio

# Together AI
pip install together

# Vertex AI
pip install google-cloud-aiplatform google-cloud-storage

# Unsloth (for local fine-tuning)
pip install unsloth transformers datasets torch trl
```

## Usage Examples

### 1. Single Fine-tuning Job (Experiment 1)

Using the `finetune_from_file` function in `utils.py`:

```python
from openai import OpenAI
from src.utils import finetune_from_file

# OpenAI (backward compatible)
client = OpenAI()
response = finetune_from_file(
    client=client,
    file_path="data/training.jsonl",
    model="gpt-4o-mini-2024-07-18",
    suffix="my-custom-model",
    provider="openai",  # default
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 8,
        "learning_rate_multiplier": 2
    }
)

# Together AI
result = finetune_from_file(
    client=None,  # Not needed for Together
    file_path="data/training.jsonl",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    suffix="my-llama-model",
    provider="together",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate": 1e-5,
        "lora": True,
        "batch_size": 8
    }
)

# Vertex AI (Google Gemini)
result = finetune_from_file(
    client=None,
    file_path="gs://my-bucket/training.jsonl",  # Can use GCS path
    model="gemini-2.0-flash-001",
    suffix="my-gemini-model",
    provider="vertex",
    hyperparameters={
        "epochs": 5,
        "adapter_size": 16,
        "learning_rate_multiplier": 1.0
    }
)

# Unsloth (Local)
result = finetune_from_file(
    client=None,
    file_path="data/training.jsonl",
    model="llama-3.1-8b-unsloth-bnb-4bit",  # Will be prefixed with unsloth/
    suffix="my-local-model",
    provider="unsloth",
    hyperparameters={
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "use_lora": True,
        "lora_r": 16,
        "learning_rate": 2e-4,
        "max_steps": 60
    }
)
```

### 2. Iterative Fine-tuning (Experiment 2 - Expert Iteration)

Using the finetuner classes in `finetuners.py`:

```python
from src.finetuners import UnifiedFinetuner, TogetherAIFinetuner

# Using the unified finetuner
finetuner = UnifiedFinetuner(
    provider="together",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate": 1e-5
    }
)

# Or use provider-specific class
finetuner = TogetherAIFinetuner(
    hyperparameters={
        "n_epochs": 3,
        "learning_rate": 1e-5,
        "lora": True
    }
)

# Use in expert iteration
result_files, model_name = await finetuner.run(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    input_log=training_log,
    log_dir="logs/experiment",
    suffix="iteration-1"
)
```

### 3. Direct Provider Usage

For more control, use the providers directly:

```python
from src.finetuning_providers import FinetuneConfig, get_finetuning_provider, Provider
import asyncio

config = FinetuneConfig(
    provider=Provider.TOGETHER,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    training_file="data/training.jsonl",
    validation_file="data/validation.jsonl",  # Optional
    suffix="custom-model",
    hyperparameters={
        "n_epochs": 3,
        "n_evals": 10,  # Evaluate 10 times during training
        "learning_rate": 1e-5,
        "lora": True,
        "batch_size": 8
    }
)

provider = get_finetuning_provider(config)
result = asyncio.run(provider.run())

print(f"Job ID: {result.job_id}")
print(f"Model Name: {result.model_name}")
print(f"Status: {result.status}")
```

## Provider-Specific Notes

### OpenAI
- Models: `gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06`, etc.
- Hyperparameters: `n_epochs`, `batch_size`, `learning_rate_multiplier`
- Supports validation splits

### Together AI
- Models: Llama, Mistral, Qwen, and more
- Supports LoRA and full fine-tuning
- LoRA serverless inference for compatible models
- Hyperparameters: `n_epochs`, `learning_rate`, `lora`, `batch_size`, `warmup_ratio`

### Vertex AI (Google Gemini)
- Models: Gemini 2.0 Flash, Gemini 2.0 Pro, etc.
- Requires GCS bucket for data storage
- Hyperparameters: `epochs`, `adapter_size`, `learning_rate_multiplier`
- Automatically creates endpoints after training

### Unsloth
- Local fine-tuning with optimized memory usage
- Supports 4-bit, 8-bit, and 16-bit training
- Best for development and testing
- Outputs saved to local directory

## Data Format

All providers expect data in JSONL format with conversational structure:

```json
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]}
```

## Monitoring

- **OpenAI**: Check status in OpenAI dashboard or via API
- **Together AI**: Use W&B integration for detailed metrics
- **Vertex AI**: Monitor in Google Cloud Console
- **Unsloth**: Local logs and TensorBoard support

## Best Practices

1. **Start Small**: Begin with a small dataset and few epochs to test
2. **Validation Data**: Always use validation data to monitor overfitting
3. **Hyperparameter Tuning**: Start with default values, then adjust
4. **Cost Management**: 
   - Use LoRA for Together AI (cheaper and faster)
   - Consider Unsloth for development/testing
   - Monitor usage in provider dashboards

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase `timeout` in config (default: 3600 seconds)
2. **File Upload Errors**: Ensure correct format and permissions
3. **API Key Issues**: Verify environment variables are set
4. **Memory Issues (Unsloth)**: Reduce `batch_size` or use smaller model

### Provider Status Checking

```python
# Check job status for any provider
from src.finetuning_providers import get_finetuning_provider, FinetuneConfig

config = FinetuneConfig(
    provider="together",
    model="...",
    training_file="...",
)

provider = get_finetuning_provider(config)
await provider.initialize_client()
status = await provider.get_job_status("job-id-here")
print(status)
```
