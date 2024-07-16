import shlex
import json
import os
import gc
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from gpt_tldr_judge import LLMJudgeConfig, llm_judge
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams
import torch


"""
python -i examples/scripts/evals/generate_rlaif.py \
    --model_name_or_path /home/ubuntu/Development/trl/models/minimal/ppo_rlaif/40 \
    --output_path examples/scripts/minimal/evals/ppo_rlaif.csv \
    --n 1000
"""


@dataclass
class Args:
    output_path: str
    model_name_or_path: str
    model_revision: str = "main"
    judge_model: str = "gpt-3.5-turbo-0125"
    n: int = 1000


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    subprocess.run(command_list, stderr=sys.stderr, stdout=sys.stdout)


MAX_TOKENS = 400  # a very generous max token length
parser = HfArgumentParser(Args)
args = parser.parse_args_into_dataclasses()[0]
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    revision=args.model_revision,
)
raw_datasets = load_dataset("zaemyung/writing_prompts_collection", split="test")
prompts = raw_datasets["prompt"]
if args.n is not None:
    prompts = prompts[: args.n]

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=MAX_TOKENS)

# Reference model
llm = LLM(
    model="Qwen/Qwen2-1.5B-Instruct",
    revision="main",
    tokenizer_revision="main",
    tensor_parallel_size=1,
)
reference_model_outputs = llm.generate(prompts, sampling_params)

# free gpu
del llm
torch.cuda.empty_cache()
gc.collect()

# RLAIF tuned
llm = LLM(
    model=args.model_name_or_path,
    revision=args.model_revision,
    tokenizer_revision=args.model_revision,
    tensor_parallel_size=1,
)
rlaif_model_outputs = llm.generate(prompts, sampling_params)

table = defaultdict(list)

# Print the outputs.
for output, reference in zip(rlaif_model_outputs, reference_model_outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    table["prompt"].append(prompt)
    table["model_response"].append(
        generated_text.strip()
    )  # need `strip()` because of the leading space
    table["model_response_len"].append(len(output.outputs[0].token_ids))

    reference_response = reference.outputs[0].text
    table["reference_response"].append(reference_response.strip())
    table["reference_response_len"].append(len(reference.outputs[0].token_ids))

df = pd.DataFrame(table)
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
df.to_json(args.output_path)

# with open(args.output_path, "r") as f:
#     df = pd.DataFrame.from_dict(json.load(f))

print(df.head())

#####
# GPT as a judge
####
df["response0"] = df["model_response"]
df["response1"] = df["reference_response"]
judged_df = llm_judge(
    LLMJudgeConfig(
        n=args.n,
        model=args.judge_model,
    ),
    df,
)
judged_df.rename(
    columns={"response0": "model_response", "response1": "reference_response"},
    inplace=True,
)
print(judged_df["preferred"].value_counts())
# print percentage
print(judged_df["preferred"].value_counts(normalize=True))

judged_df.to_json(args.output_path[: -len(".json")] + "_judged.json")
