from src.expert_iteration import Finetuner
from typing import Tuple, Any
try:
    from typing import override
except ImportError:
    from typing_extensions import override
import os
import json
import warnings
import time
import asyncio
import concurrent.futures
from src.expert_iteration import Log


class UnifiedFinetuner(Finetuner):
    """Simple unified finetuner for all providers."""
    
    def __init__(
        self,
        provider: str = "openai",
        msg_roles_to_extract: list[str] = ["system", "user", "assistant"],
        check_status_every: int = 120,
        timeout: int = 60 * 60,
        **hyperparameters
    ):
        self.provider = provider.lower()
        self.msg_roles_to_extract = msg_roles_to_extract
        self.check_status_every = check_status_every
        self.timeout = timeout
        self.hyperparameters = hyperparameters

    @override
    async def run(
        self, model: str, input_log: Log, log_dir: str, suffix: str, **kwargs
    ) -> Tuple[Any, str]:
        # Prepare training data from input_log
        training_data = self._prepare_training_data(input_log)
        
        # Save training data to file
        training_file = os.path.join(log_dir, "training_data.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        with open(training_file, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")
        
        # Run provider-specific fine-tuning
        if self.provider == "openai":
            result = await self._run_openai(model, training_file, suffix)
        elif self.provider == "together":
            result = await self._run_together(model, training_file, suffix)
        elif self.provider == "unsloth":
            result = await self._run_unsloth(model, training_file, suffix)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Log results
        self._log_results(result, log_dir)
        
        # Check for errors
        if result["status"] == "failed":
            raise RuntimeError(f"Fine-tuning failed: {result.get('error')}")
        elif result["status"] == "cancelled":
            raise RuntimeError(f"Fine-tuning was cancelled")
        
        return result.get("provider_data"), result["model_name"]
    
    @override
    async def retry(
        self, model: str, input_log: Any, log_dir: str, suffix: str, **kwargs
    ) -> Tuple[Any, str]:
        warnings.warn("Retrying fine-tuning...")
        return await self.run(model, input_log, log_dir, suffix, **kwargs)
    
    async def _run_openai(self, model: str, training_file: str, suffix: str) -> dict:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Upload file
        with open(training_file, "rb") as f:
            file_response = await client.files.create(file=f, purpose="fine-tune")
        
        # Start fine-tuning
        job = await client.fine_tuning.jobs.create(
            training_file=file_response.id,
            model=model,
            suffix=suffix,
            hyperparameters=self.hyperparameters
        )
        
        # Wait for completion
        start_time = time.time()
        while True:
            job_status = await client.fine_tuning.jobs.retrieve(job.id)
            
            if job_status.status in ["succeeded", "failed", "cancelled"]:
                return {
                    "job_id": job.id,
                    "model_name": job_status.fine_tuned_model or "",
                    "status": job_status.status,
                    "error": str(job_status.error) if job_status.error else None,
                    "provider_data": {"result_files": job_status.result_files}
                }
            
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Job {job.id} timed out after {self.timeout}s")
            
            await asyncio.sleep(self.check_status_every)
    
    async def _run_together(self, model: str, training_file: str, suffix: str) -> dict:
        try:
            from together import Together
        except ImportError:
            raise ImportError("Please install 'together' package: pip install together")
        
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        
        # Upload file
        file_response = client.files.upload(training_file, check=True)
        
        # Prepare parameters
        params = {
            "training_file": file_response.id,
            "model": model,
            "suffix": suffix,
            "n_epochs": self.hyperparameters.get("n_epochs", 3),
            "learning_rate": self.hyperparameters.get("learning_rate", 1e-5),
            "batch_size": self.hyperparameters.get("batch_size", "max"),
        }
        
        # Start fine-tuning
        job = client.fine_tuning.create(**params)
        
        # Wait for completion
        start_time = time.time()
        while True:
            job_status = client.fine_tuning.retrieve(job.id)
            
            if job_status.status == "completed":
                return {
                    "job_id": job.id,
                    "model_name": job_status.output_name,
                    "status": "succeeded",
                    "error": None,
                    "provider_data": {"job": job_status}
                }
            elif job_status.status in ["failed", "cancelled"]:
                return {
                    "job_id": job.id,
                    "model_name": "",
                    "status": job_status.status,
                    "error": f"Job {job.id} {job_status.status}",
                    "provider_data": None
                }
            
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Job {job.id} timed out after {self.timeout}s")
            
            await asyncio.sleep(self.check_status_every)
    
    async def _run_unsloth(self, model: str, training_file: str, suffix: str) -> dict:
        # Run in thread pool since Unsloth is synchronous
        def _train():
            try:
                from unsloth import FastLanguageModel
                from datasets import load_dataset
                from transformers import TrainingArguments
                from trl import SFTTrainer
            except ImportError as e:
                raise ImportError(f"Please install required packages for Unsloth: {e}")
            
            # Load model
            model_name = model if model.startswith("unsloth/") else f"unsloth/{model}"
            model_obj, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.hyperparameters.get("max_seq_length", 2048),
                load_in_4bit=self.hyperparameters.get("load_in_4bit", True),
            )
            
            # Apply LoRA
            if self.hyperparameters.get("use_lora", True):
                model_obj = FastLanguageModel.get_peft_model(
                    model_obj,
                    r=self.hyperparameters.get("lora_r", 16),
                    lora_alpha=self.hyperparameters.get("lora_alpha", 16),
                    lora_dropout=self.hyperparameters.get("lora_dropout", 0.1),
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                )
            
            # Load dataset and train
            dataset = load_dataset("json", data_files=training_file, split="train")
            
            training_args = TrainingArguments(
                output_dir=f"./outputs/{suffix}",
                per_device_train_batch_size=self.hyperparameters.get("batch_size", 2),
                max_steps=self.hyperparameters.get("max_steps", 60),
                learning_rate=self.hyperparameters.get("learning_rate", 2e-4),
                logging_steps=1,
            )
            
            trainer = SFTTrainer(
                model=model_obj,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=training_args,
                dataset_text_field="text",
                max_seq_length=self.hyperparameters.get("max_seq_length", 2048),
            )
            
            trainer.train()
            
            # Save model
            output_dir = f"./outputs/{suffix}_final"
            model_obj.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            return output_dir
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_train)
            try:
                output_dir = await asyncio.get_event_loop().run_in_executor(None, future.result, self.timeout)
                return {
                    "job_id": f"unsloth_{int(time.time())}",
                    "model_name": output_dir,
                    "status": "succeeded",
                    "error": None,
                    "provider_data": {"output_dir": output_dir}
                }
            except Exception as e:
                return {
                    "job_id": f"unsloth_{int(time.time())}",
                    "model_name": "",
                    "status": "failed",
                    "error": str(e),
                    "provider_data": None
                }
    
    def _prepare_training_data(self, input_log: Log) -> list:
        training_data = []
        
        # Handle different input log formats
        if isinstance(input_log, str):
            # String format - split by lines and parse JSON
            for line in input_log.strip().split('\n'):
                item = json.loads(line)
                processed = self._process_training_item(item)
                if processed:
                    training_data.append(processed)
        elif isinstance(input_log, list):
            # List format - process each item directly
            for item in input_log:
                processed = self._process_training_item(item)
                if processed:
                    training_data.append(processed)
        else:
            # Other formats - try to extract conversations
            if hasattr(input_log, 'conversations'):
                for conv in input_log.conversations:
                    if hasattr(conv, 'messages'):
                        item = {"messages": conv.messages}
                    else:
                        item = {"messages": conv}
                    processed = self._process_training_item(item)
                    if processed:
                        training_data.append(processed)
            else:
                raise ValueError(f"Unsupported input_log format: {type(input_log)}")
        
        return training_data
    
    def _process_training_item(self, item: dict) -> dict:
        messages = item.get("messages", [])
        
        # Filter messages by role
        filtered_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in self.msg_roles_to_extract
        ]

        # Add empty system message if missing
        if not any(msg["role"] == "system" for msg in filtered_messages):
            filtered_messages.insert(0, {"role": "system", "content": " "})

        # Validate we have user and assistant messages
        if (len(filtered_messages) >= 2 and 
            any(msg["role"] == "user" for msg in filtered_messages) and
            any(msg["role"] == "assistant" for msg in filtered_messages)):
            return {"messages": filtered_messages}
        
        # Return empty if validation fails
        return None
    
    def _log_results(self, result: dict, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "fine_tuning_results.json"), "w") as f:
            json.dump(result, f, indent=2)