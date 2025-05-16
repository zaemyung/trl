#!/usr/bin/env python3
# -------------------------------------------------
# Elo tournament for LLM generations – Anthropics Batch edition
# -------------------------------------------------
import json
import os
import random
import sys
import time
from glob import glob
from pprint import pprint
from typing import List

import anthropic
import matplotlib.pyplot as plt
import regex as re
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv
from natsort import natsorted
from rich.progress import track


# ─────────────────────────── set-up ────────────────────────────
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

exp_name = sys.argv[1]  # e.g. "mpo_variations"
num_matches = 2000  # total pairwise judgements
judge_model = "claude-3-7-sonnet-20250219"
batch_size = 100  # 100 prompts per Claude batch

# ─────────────────── model alias map per experiment ────────────
if exp_name == "rm_32":
    model_names_to_annon = {
        "32b_72b": "ModelA",
        "autoprompt-32b": "ModelB",
        "expert-32b": "ModelC",
        "base-1.5b": "ModelD",
        "iter0-32b": "ModelE",
    }
elif exp_name == "rm_72":
    model_names_to_annon = {
        "72b_72b": "ModelA",
        "autoprompt-72b": "ModelB",
        "expert-72b": "ModelC",
        "base-1.5b": "ModelD",
        "iter0-72b": "ModelE",
    }
elif exp_name == "mpo_variations":
    model_names_to_annon = {
        "32b_32b": "ModelA",
        "32b_72b": "ModelB",
        "72b_32b": "ModelC",
        "72b_72b": "ModelD",
    }
elif exp_name == "32b_32bvs32b_72b":
    model_names_to_annon = {"32b_32b": "ModelA", "32b_72b": "ModelB"}
elif exp_name == "72b_32bvs72b_72b":
    model_names_to_annon = {"72b_32b": "ModelA", "72b_72b": "ModelB"}
elif exp_name == "mpo_vs_oracle":
    model_names_to_annon = {
        "32b_72b": "ModelB",
        "72b_72b": "ModelD",
        "oracle-32b": "ModelI",
        "oracle-72b": "ModelJ",
    }
elif exp_name == "summarization":
    model_names_to_annon = {
        "32b_32b": "ModelA",
        "autoprompt-32b": "ModelB",
        "iter0-32b": "ModelC",
        "base": "ModelD",
    }
else:
    raise ValueError(f"Unknown exp_name: {exp_name}")

annon_to_model_names = {v: k for k, v in model_names_to_annon.items()}

# ───────────────────────── utilities ───────────────────────────


def load_data(jsonl_paths, separation_regex):
    """Read .jsonl files and split prompt/response pairs."""
    data_dict = {}
    for model_name, path in jsonl_paths.items():
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                m = re.search(separation_regex, obj["query"], re.MULTILINE | re.DOTALL)
                query = m.group(2).strip()
                entries.append((query, obj["model_response"]))
        data_dict[model_names_to_annon[model_name]] = entries
    return data_dict


def update_elo(r_a, r_b, score_a, *, k=32, base=10, scale=400):
    e_a = 1 / (1 + base ** ((r_b - r_a) / scale))
    e_b = 1 - e_a
    return (
        r_a + k * (score_a - e_a),
        r_b + k * ((1 - score_a) - e_b),
    )


# ─────────────────── Anthropic batch helpers ───────────────────
MAX_TOKENS = 32
POLL_SEC = 2


def call_claude_batch(
    prompts: List[str],
    *,
    system_prompt: str,
    model: str = judge_model,
    temperature: float = 0.0,
):
    """Single Messages-Batch request – returns replies in original order."""
    reqs = [
        Request(
            custom_id=str(i),
            params={
                "model": model,
                "max_tokens": MAX_TOKENS,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": p}],
                "stop_sequences": ["<EOE>"],
            },
        )
        for i, p in enumerate(prompts)
    ]

    batch = client.beta.messages.batches.create(requests=reqs)
    bid = batch.id
    while batch.processing_status != "ended":
        time.sleep(POLL_SEC)
        batch = client.beta.messages.batches.retrieve(bid)

    out = {}
    for res in client.beta.messages.batches.results(bid):
        if res.result.type == "succeeded":
            txt = "".join(block.text for block in res.result.message.content)
            out[res.custom_id] = txt
        else:
            out[res.custom_id] = ""
    return [out[str(i)] for i in range(len(prompts))]


# ─────────────── prompt builders & batch judgement ─────────────


def build_prompt_essay(q, a_name, a_resp, b_name, b_resp):
    return f"""Writing Prompt:
{q}

Response from {a_name}:
{a_resp}

Response from {b_name}:
{b_resp}

Which response is better overall? Reply with '<winner>{a_name}</winner>' or '<winner>{b_name}</winner>' followed by <EOE>."""


def build_prompt_summary(q, a_name, a_resp, b_name, b_resp):
    return f"""Government Bill:
{q}

Summary from {a_name}:
{a_resp}

Summary from {b_name}:
{b_resp}

Which summary is better overall? Reply with '<winner>{a_name}</winner>' or '<winner>{b_name}</winner>' followed by <EOE>."""


def judge_responses_batch(task, pairs, *, model=judge_model):
    if task == "essay_writing":
        build = build_prompt_essay
        sys_prompt = (
            "You are an impartial and insightful judge. Evaluate which essay responds "
            "better, considering clarity, coherence, depth of argument, and effectiveness."
        )
    else:  # summarization
        build = build_prompt_summary
        sys_prompt = (
            "You are an impartial and discerning evaluator. Assess which summary best "
            "represents the bill, weighing conciseness, accuracy, and faithfulness."
        )

    prompts = [build(p["query"], p["a_name"], p["a_resp"], p["b_name"], p["b_resp"]) for p in pairs]
    replies = call_claude_batch(prompts, system_prompt=sys_prompt, model=model)

    winners = []
    rgx = re.compile(r"<winner>(.+?)</winner>", re.I | re.S)
    for rep, p in zip(replies, pairs):
        m = rgx.search(rep)
        if not m:
            winners.append(None)
            continue
        tag = m.group(1).strip().lower()
        if p["a_name"].lower() in tag:
            winners.append(p["a_name"])
        elif p["b_name"].lower() in tag:
            winners.append(p["b_name"])
        else:
            winners.append(None)
    return winners


# ───────────────────── main Elo simulation ─────────────────────


def run_elo_simulation_batch(
    task,
    data_dict,
    *,
    n_matches,
    k_factor=4,
    batch_size=batch_size,
    model=judge_model,
):
    elo = {m: 1000 for m in data_dict}
    names = list(data_dict)
    buf = []
    results = []

    def flush():
        nonlocal buf
        if not buf:
            return
        winners = judge_responses_batch(task, buf, model=model)
        for match, win in zip(buf, winners):
            if win is None:
                continue
            a, b = match["a_name"], match["b_name"]
            score_a = 1 if win == a else 0
            elo[a], elo[b] = update_elo(elo[a], elo[b], score_a, k=k_factor)
            results.append(
                {
                    "query": match["query"],
                    "model_a": annon_to_model_names[a],
                    "model_b": annon_to_model_names[b],
                    "winner": annon_to_model_names[win],
                    "model_a_response": match["a_resp"],
                    "model_b_response": match["b_resp"],
                }
            )
        buf = []

    for _ in track(range(n_matches), description="Judging…", total=n_matches):
        a, b = random.sample(names, 2)
        idx = random.randrange(len(data_dict[a]))
        q, ra = data_dict[a][idx]
        _, rb = data_dict[b][idx]

        buf.append(dict(query=q, a_name=a, a_resp=ra, b_name=b, b_resp=rb))
        if len(buf) >= batch_size:
            flush()
    flush()
    return elo, results


# ─────────────────────────── I/O helpers ───────────────────────


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_scores(scores, path):
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels, vals = zip(*ordered)
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 6))
    plt.bar(labels, vals)
    plt.ylabel("Elo")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_sim(
    task_name,
    iteration,
    *,
    n_matches,
    gen_dir,
    out_dir,
    separation_regex,
):
    print(f"{task_name}: iteration {iteration}")
    paths = {k: f"{gen_dir}/{k}.test.generations.jsonl" for k in model_names_to_annon}
    data = load_data(paths, separation_regex)
    elo, details = run_elo_simulation_batch(
        task_name,
        data,
        n_matches=n_matches,
        model=judge_model,
    )
    elo = {annon_to_model_names[k]: v for k, v in elo.items()}

    os.makedirs(out_dir, exist_ok=True)
    score_path = f"{out_dir}/{exp_name}-{iteration}.scores.json"
    detail_path = f"{out_dir}/{exp_name}-{iteration}.details.json"
    save_json(elo, score_path)
    save_json(details, detail_path)
    plot_scores(elo, score_path.replace(".json", ".pdf"))


def merge_results(out_dir, exp):
    d_paths = natsorted(glob(os.path.join(out_dir, f"{exp}-*.details.json")))
    s_paths = natsorted(glob(os.path.join(out_dir, f"{exp}-*.scores.json")))
    all_details = []
    for p in d_paths:
        with open(p) as f:
            all_details.extend(json.load(f))
    all_scores = [json.load(open(p)) for p in s_paths]

    mean_scores = {}
    std_scores = {}
    for k in all_scores[0]:
        vals = [s[k] for s in all_scores]
        mean_scores[k] = sum(vals) / len(vals)
        std_scores[k] = (sum((v - mean_scores[k]) ** 2 for v in vals) / len(vals)) ** 0.5

    save_json(all_details, f"{out_dir}/{exp}-merged.details.json")
    save_json(
        {"mean elo": mean_scores, "mean std": std_scores},
        f"{out_dir}/{exp}-merged.scores.json",
    )
    pprint({"mean elo": mean_scores, "mean std": std_scores})


# ────────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    task = "summarization"  # or "essay_writing"
    gen_dir = f"results/policy-1.5b/generations/{task}"
    out_dir = f"results/policy-1.5b/elo_scores/{task}"

    if task == "essay_writing":
        sep_rx = r"user(.+?)Instructions:(.+?)Your Writing:"
    else:
        sep_rx = r"You are a helpful assistant\.\nuser(.+?)Bill:\n```(.+?)```"

    for i in range(1, 6):
        run_sim(
            task,
            i,
            n_matches=num_matches,
            gen_dir=gen_dir,
            out_dir=out_dir,
            separation_regex=sep_rx,
        )
    merge_results(out_dir, exp_name)
