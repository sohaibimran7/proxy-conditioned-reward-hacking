from src.expert_iteration import Sampler, Log
from src.inspect_helpers.sampler import EvalSampler
from typing import Any, Optional, List, Tuple
import pandas as pd
import json
import os
from typing import Dict
try:
    from typing import override
except ImportError:
    from typing_extensions import override
from pydantic import BaseModel

SAMPLE_FILE = "samples.json"


class SampleLog(BaseModel):
    messages: List[Dict[str, Any]]
    score: float
    ascending: bool
    conditions: List[Tuple[str, Any]] | List[Tuple[str, str, Any]] | None = None


class InspectSampler(Sampler):
    def __init__(
        self,
        rank_column: str,
        n: int = 10,
        ascending: bool = False,
        conditions: List[Tuple[str, Any]] | List[Tuple[str, str, Any]] | None = None,
    ):
        self.rank_column = rank_column
        self.n = n
        self.ascending = ascending
        self.conditions = conditions

    @override
    async def run(self, input_log: Log, log_dir: str, **irrelevant) -> Log:
        # Assert input log is instance of BaseModel or BaseModel list of len 1
        assert (
            isinstance(input_log, BaseModel)
            or (
                isinstance(input_log, list)
                and len(input_log) == 1
                and isinstance(input_log[0], BaseModel)
            )
        ), "input_log must be an instance of BaseModel or a list containing a single BaseModel instance"

        if isinstance(input_log, list):
            input_log = input_log[0]

        sampler = EvalSampler(input_log)
        samples_df = sampler.rank_samples(
            rank_column=self.rank_column,
            n=self.n,
            ascending=self.ascending,
            conditions=self.conditions,
        )
        log = await self.make_log(samples_df)
        await self.save_log(log, log_dir)
        return log

    async def make_log(self, samples_df: pd.DataFrame) -> str:
        log_data = []
        for _, row in samples_df.iterrows():
            messages = row["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)

            log_entry = SampleLog(
                messages=messages,
                score=row[self.rank_column],
                ascending=self.ascending,
                conditions=self.conditions,
            )
            log_data.append(log_entry.model_dump_json())

        return "\n".join(log_data)  # Join entries with newlines for JSONL format

    async def save_log(self, log: str, log_dir: str) -> str:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, SAMPLE_FILE), "w") as f:
            f.write(log)
