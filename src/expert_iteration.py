from inspect_ai.log import EvalLog
from pydantic import BaseModel, Field
from typing import Union, Dict, List, Any
from abc import ABC, abstractmethod
from typing import Callable, Tuple
try:
    from typing import override
except ImportError:
    from typing_extensions import override
import ast
import json
import pickle
import os
import warnings
import aiofiles
import dill

SUCCESS = "success"
FAILURE = "failed"

LOG_FILE = "expert_iteration_log.json"
DILL_FILE = "expert_iteration_instance.dill"

Log = str | BaseModel | list[BaseModel] | list[dict]

class AbstractStage(ABC):
    @abstractmethod
    async def run(self, *args, **kwargs) -> Log:
        pass

    async def retry(self, *args, **kwargs) -> Log:
        return await self.run(*args, **kwargs)

class Evaluator(AbstractStage):
    @override
    @abstractmethod
    async def run(self, modelprovider: str | None, model: str, log_dir: str, **irrelevant) -> Log:
        pass

class Sampler(AbstractStage):
    @override
    @abstractmethod
    async def run(self, input_log: Log, log_dir: str, **irrelevant) -> Log:
        pass

class Finetuner(AbstractStage):
    @override
    @abstractmethod
    # for local models the return is a tuple of (log, model_path) instead of (log, model_name)
    async def run(self, model: str, input_log: Log, log_dir: str, suffix: str, **irrelevant) -> Tuple[Log, str]:
        pass

class RetryConfig(BaseModel):
    evaluation: int = 0
    sampling: int = 0
    finetuning: int = 0

class ExpertIterationConfig(BaseModel):
    max_iter: int
    modelprovider: str
    model: str
    log_dir: str
    retries: Union[int, RetryConfig] = Field(default=RetryConfig())
    suffix: str = ""

class AttemptLog(BaseModel):
    status: str
    error: str | None = None

class StageLog(BaseModel):
    stage: str
    status: str | None = None
    attempts: List[AttemptLog] = Field(default_factory=list)
    error: str | None = None
    log_dir: str | None = None

class IterationLog(BaseModel):
    iter: int
    input_model: str
    output_model: str | None = None
    stages: List[StageLog] = Field(default_factory=list)
    status: str | None = None

class ExpertIterationLog(BaseModel):
    status: str | None = None
    model: str | None = None
    config: ExpertIterationConfig
    iterations: List[IterationLog] = Field(default_factory=list)

class ExpertIteration:
    def __init__(self, config: ExpertIterationConfig, evaluator: Evaluator, sampler: Sampler, finetuner: Finetuner):
        self.config = config
        self.model = config.model
        self.evaluator = evaluator
        self.sampler = sampler
        self.finetuner = finetuner
        self.current_iter : int = 0
        self.current_stage : str = None
        self.last_log : Log | None = None
        self.is_retry : bool = False
        self.log : ExpertIterationLog = ExpertIterationLog(config=config)

    async def run(self):
        while self.current_iter < self.config.max_iter:
            try:
                await self._run_iteration()
                self.current_iter += 1
            except Exception as e:
                warnings.warn(f"Iteration {self.current_iter} failed: {str(e)}")
                self._write_log()
                await self.save_state()
                raise
        
        await self.save_state()
        self._log_expert_iteration_success()
        self._write_log()

    async def _run_iteration(self):
        stage_map = {
            "evaluation": self.evaluator,
            "sampling": self.sampler,
            "finetuning": self.finetuner
        }
        self._log_iteration_start()

        for stage in stage_map:

            self.current_stage = stage
            self._log_stage_start()

            stage_obj = stage_map[stage]

            # Automatic retries
            retries = self.config.retries
            if isinstance(retries, RetryConfig):
                retries = getattr(retries, self.current_stage)
        
            result = await self._try_stage(stage_obj, remaining_retries=retries)
            self.is_retry = False

            if isinstance(stage_obj, Finetuner):
                self.last_log, self.model = result
            else:
                self.last_log = result

        self._log_iteration_success()
        self.current_stage = None

    async def _try_stage(self, stage_obj: AbstractStage, remaining_retries: int | None = None) -> Union[Log, Tuple[Log, str]]:
        try:
            result = await self._run_stage(stage_obj)
            self._log_attempt_success()
            return result
        except Exception as e:
            self._log_attempt_failure(str(e))
            if remaining_retries > 0:
                self.is_retry = True
                warnings.warn(f"Error {e} encountered while running stage {self.current_stage} of iteration {self.current_iter}. Retrying {remaining_retries} more times")
                return await self._try_stage(stage_obj, remaining_retries - 1)
            else:
                self._log_stage_failure(str(e))
                raise

    async def _run_stage(self, stage_obj) -> Union[Log, Tuple[Log, str]]:
        kwargs = {
            "modelprovider": self.config.modelprovider,
            "model": self.model,
            "input_log": self.last_log,
            "log_dir": self._get_log_dir(),
            "suffix": self._get_suffix()
        }
        if self.is_retry:
            return await stage_obj.retry(**kwargs)
        return await stage_obj.run(**kwargs)

    def _get_log_dir(self) -> str:
        return os.path.join(self.config.log_dir, f"iter_{self.current_iter}", self.current_stage)

    def _get_suffix(self) -> str:
        return f"{self.config.suffix}_iter_{self.current_iter+1}"

    def _log_attempt_success(self):
        stage_log = self._get_stage_log()
        stage_log.attempts.append(AttemptLog(status=SUCCESS))
        self._log_stage_success()

    def _log_attempt_failure(self, error: str):
        stage_log = self._get_stage_log()
        stage_log.attempts.append(AttemptLog(status=FAILURE, error=error))

    def _log_stage_start(self):
        stage_log = StageLog(stage=self.current_stage, log_dir=self._get_log_dir())
        self.log.iterations[-1].stages.append(stage_log)

    def _log_stage_success(self):
        stage_log = self._get_stage_log()
        stage_log.status = SUCCESS

    def _log_stage_failure(self, error: str):
        stage_log = self._get_stage_log()
        stage_log.status = FAILURE
        stage_log.error = error
        self._log_iteration_failure()

    def _log_iteration_start(self):
        self.log.iterations.append(IterationLog(iter=self.current_iter, input_model=self.model, stages=[], status=None))

    def _log_iteration_success(self):
        self.log.iterations[-1].status = "success"
        self.log.iterations[-1].output_model = self.model

    def _log_iteration_failure(self):
        self.log.iterations[-1].status = FAILURE
        self._log_expert_iteration_failure()
        
    def _log_expert_iteration_success(self):
        self.log.status = SUCCESS
        self.log.model = self.model

    def _log_expert_iteration_failure(self):
        self.log.status = FAILURE
        self.log.model = self.model

    def _write_log(self):
        os.makedirs(self.config.log_dir, exist_ok=True)
        with open(os.path.join(self.config.log_dir, LOG_FILE), "w") as f:
            f.write(self.log.model_dump_json())

    def _get_stage_log(self) -> StageLog:
        return self.log.iterations[-1].stages[-1]

    #Manual retry, retries the entire expert iteration
    async def retry(self):
        self.is_retry = True
        last_iteration = self.log.iterations[-1]
        self.current_iter = last_iteration.iter
        last_stage_log = last_iteration.stages[-1]
        
        if last_stage_log.status == "failed":
            self.current_stage = last_stage_log.stage
            warnings.warn(f"Retrying expert iteration from iteration {self.current_iter}, stage: {self.current_stage}")
            # delete the last iteration log 
            del self.log.iterations[-1]
            await self.run()
        else:
            warnings.warn("No failed stage found to retry.")

    async def _run_for_more_iterations(self, max_iter: int):
        if max_iter <= self.config.max_iter:
            warnings.warn(f"Expert iteration was already run for {self.config.max_iter} iterations. Perhaps you want to retry the expert iteration instead?")
        else:
            self.config.max_iter = max_iter
            await self.run()


    async def save_state(self):
        """Asynchronously saves the expert_iteration to log_dir."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        async with aiofiles.open(os.path.join(self.config.log_dir, DILL_FILE), "wb") as f:
            await f.write(dill.dumps(self))

    @staticmethod
    async def load_state(filename: str) -> 'ExpertIteration':
        """Asynchronously loads the state from a file."""
        async with aiofiles.open(filename, "rb") as f:
            data = await f.read()
            return dill.loads(data)
