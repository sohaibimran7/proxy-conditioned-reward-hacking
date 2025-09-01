from src.expert_iteration import Evaluator
from inspect_ai import eval_async, eval_retry_async, Tasks, Epochs
from inspect_ai.solver import Plan, Solver
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_ai.model import (
    GenerateConfigArgs,
    Model,
)
from inspect_ai.log import list_eval_logs, read_eval_log
from typing import Any
try:
    from typing import override
except ImportError:
    from typing_extensions import override
from typing_extensions import Unpack
from datetime import datetime, timezone
import warnings
import asyncio

class InspectEvaluator(Evaluator):
    def __init__(
        self,
        tasks: Tasks,
        model: str | Model | list[str] | list[Model] | None = None,
        model_args: dict[str, Any] = dict(),
        task_args: dict[str, Any] = dict(),
        sandbox: SandboxEnvironmentSpec | None = None,
        sandbox_cleanup: bool | None = None,
        plan: Plan | Solver | list[Solver] | None = None,
        log_level: str | None = None,
        limit: int | tuple[int, int] | None = None,
        epochs: int | Epochs | None = None,
        max_messages: int | None = None,
        max_samples: int | None = None,
        max_tasks: int | None = None,
        max_subprocesses: int | None = None,
        log_samples: bool | None = None,
        log_images: bool | None = None,
        log_buffer: int | None = None,
        score: bool = True,
        **generate_config_kwargs: Unpack[GenerateConfigArgs],
    ):
        super().__init__()
        self.tasks = tasks
        self.model_args = model_args
        self.task_args = task_args
        self.sandbox = sandbox
        self.sandbox_cleanup = sandbox_cleanup
        self.plan = plan
        self.log_level = log_level
        self.limit = limit
        self.epochs = epochs
        self.max_messages = max_messages
        self.max_samples = max_samples
        self.max_tasks = max_tasks
        self.max_subprocesses = max_subprocesses
        self.log_samples = log_samples
        self.log_images = log_images
        self.log_buffer = log_buffer
        self.score = score
        self.generate_config_kwargs = generate_config_kwargs
        self.last_run_time = None

    async def _execute_eval(self, eval_func, *args, **kwargs):
        while True:
            try:
                logs = await eval_func(*args, **kwargs)
                if isinstance(logs, list):
                    log = logs[0]
                if log.status == "error":
                    raise RuntimeError(f"Evaluation failed with error: {log.error}")
                return log
            except RuntimeError as e:
                if str(e) == "Multiple concurrent calls to eval_async are not allowed.":
                    warnings.warn("Concurrent eval_async call detected. Retrying...")
                    await asyncio.sleep(1)  # Wait a bit before retrying
                else:
                    raise  # Re-raise if it's a different RuntimeError

    @override
    async def run(self, modelprovider: str, model: str | Model, log_dir: str, **irrelevant) -> str:
        if 'local' in modelprovider:
            self.model_args['model_path'] = model
            model = modelprovider
        else:
            model = f"{modelprovider}/{model}"

        self.last_run_time = datetime.now(timezone.utc)
        
        return await self._execute_eval(
            eval_async,
            model=model,
            log_dir=log_dir,
            model_args=self.model_args,
            tasks=self.tasks,
            task_args=self.task_args,
            sandbox=self.sandbox,
            sandbox_cleanup=self.sandbox_cleanup,
            plan=self.plan,
            log_level=self.log_level,
            limit=self.limit,
            epochs=self.epochs,
            max_messages=self.max_messages,
            max_samples=self.max_samples,
            max_tasks=self.max_tasks,
            max_subprocesses=self.max_subprocesses,
            log_samples=self.log_samples,
            log_images=self.log_images,
            log_buffer=self.log_buffer,
            score=self.score,
            **self.generate_config_kwargs,
        )

    @override
    async def retry(self, modelprovider: str | None, model: str | Model, log_dir: str, **irrelevant) -> str:
        latest_log = read_eval_log(list_eval_logs(log_dir, recursive=False)[-1])
        log_created_time = datetime.fromisoformat(latest_log.eval.created)

        if self.last_run_time is None or log_created_time <= self.last_run_time:
            warnings.warn("No new logs saved since last run. Rerunning evaluation from scratch again.")
            return await self.run(modelprovider, model, log_dir, **irrelevant)
        
        return await self._execute_eval(
            eval_retry_async,
            latest_log,
            log_dir=log_dir,
            log_level=self.log_level,
            max_samples=self.max_samples,
            max_tasks=self.max_tasks,
            max_subprocesses=self.max_subprocesses,
            sandbox_cleanup=self.sandbox_cleanup,
            log_samples=self.log_samples,
            log_images=self.log_images,
            log_buffer=self.log_buffer,
            score=self.score,
            **self.generate_config_kwargs,
        )
