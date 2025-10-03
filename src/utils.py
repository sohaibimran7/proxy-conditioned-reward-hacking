from src.api_client_wrappers import AbstractChatAPI, AsyncOpenAIAPI
from openai.types.fine_tuning import FineTuningJob
from openai import OpenAI
from together import Together
# from pprint import pprint  # Not needed for simplified API
import json
import random


def read_file(filename: str) -> str:
    with open(filename, 'r') as f:
        return f.read()
    
def write_to_file(content: str, filename: str) -> None:
    with open(filename, 'w') as f:
        f.write(content)
    
def readlines(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        return f.readlines()

def read_jsonl_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()

def shuffle_jsonl(input_file, output_file):
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")



def get_model_names_to_evaluate(
    provider: str = "openai",
    base_models: list[str] = [],
    suffix: str = None,
    include_base_models: bool = True,
):
    finetunes = get_finetuning_jobs(provider, base_models, suffix)

    models = base_models if include_base_models else []
    for job in finetunes:
        if provider == "openai":
            models.append(job.fine_tuned_model)
        elif provider == "together":
            models.append(job.output_name)

    return [f"{provider}/{model}" for model in models]

def get_finetuning_jobs(
    provider: str,
    base_models: list[str],
    suffix: str,
    status: str = "succeeded",
) -> list[FineTuningJob]:
    if provider == "openai":
        client = OpenAI()
        ft_jobs = client.fine_tuning.jobs.list()
        
        return [
            job
            for job in ft_jobs
            if job.user_provided_suffix == suffix
            and job.model in base_models
            and job.status == status
        ]
    
    elif provider == "together":
        client = Together()
        ft_list = client.fine_tuning.list()
        ft_jobs = ft_list.data
        
        status = "completed" if status == "succeeded" else status
        
        return [
            job
            for job in ft_jobs
            if job.suffix == suffix
            and job.model in base_models
            and job.status == status
        ]
    else:
        raise NotImplementedError(f"Client {provider} not supported")


def get_finetuning_jobs_from_substrings(
    client,
    suffix_substring: str,
    exclude_suffixes: list[str],
    base_model_substrings: list[str],
    status: str = "succeeded",
) -> list[FineTuningJob]:
    ft_jobs = client.fine_tuning.jobs.list()
    
    return [
        job
        for job in ft_jobs
        if job.user_provided_suffix
        and suffix_substring in job.user_provided_suffix
        and any(
            base_model_substring in job.model
            for base_model_substring in base_model_substrings
        )
        and not any(
            exclude_substring in job.user_provided_suffix
            for exclude_substring in exclude_suffixes
        )
        and job.status == status
    ]


def get_checkpoint_models(
    client: OpenAI, job: FineTuningJob, checkpoints_to_evaluate: list[int]
) -> list[str]:
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job.id).data
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x.step_number)

    selected_checkpoints = []
    for index in checkpoints_to_evaluate:
        if 0 <= index < len(sorted_checkpoints):
            selected_checkpoints.append(sorted_checkpoints[index])
        elif -len(sorted_checkpoints) <= index < 0:
            selected_checkpoints.append(sorted_checkpoints[index])

    return [
        checkpoint.fine_tuned_model_checkpoint for checkpoint in selected_checkpoints
    ]


# make model dynamic
english_translator_obj = AsyncOpenAIAPI(
    model="gpt-4o-mini",
    system_prompt="Please translate the provided text into English",
    response_format=None,
    temperature=0,
)


async def translator(
    text: str, translator_obj: AbstractChatAPI = english_translator_obj
) -> str:
    return await translator_obj.generate_response(text)
