import os
import random
import shutil

import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, MPOConfig, MPOTrainer, ScriptArguments, get_kbit_device_map, get_quantization_config
from trl.extras.mpo import get_task_dataset
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file examples/accelerate_configs/single_gpu.yaml \
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file examples/accelerate_configs/multi_gpu_6gpus.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2_6gpus.yaml \
    examples/scripts/mpo.py \
    --dataset_name "essay_writing" \
    --task_name "essay_writing" \
    --wandb_entity "iterater" \
    --wandb_project "mpo-new" \
    --exp_name "essay_writing-mpo-32b_32b" \
    --output_dir models/mpo_essay_writing \
    --learning_rate 3e-6 \
    --num_ppo_epochs 4 \
    --num_mpo_interval 10 \
    --num_mpo_samples 20 \
    --num_mini_batches 2 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --local_rollout_forward_batch_size 1 \
    --total_episodes 26013 \
    --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --sft_model_path Qwen/Qwen2-1.5B-Instruct \
    --response_length 400 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.02 \
    --stop_token eos \
    --reward_model_address "http://129.213.31.51:30000" \
    --meta_reward_model_address "http://129.213.31.51:30000"
"""


def seed_everything(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything(42)
    parser = HfArgumentParser((ScriptArguments, MPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # if os.path.exists(training_args.output_dir):
    #     raise ValueError(
    #         f"Output directory ({training_args.output_dir}) already exists. Please choose a different output directory."
    #     )
    # else:
    #     shutil.rmtree(training_args.output_dir, ignore_errors=True)
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    task_name = training_args.task_name
    assert task_name in [
        "essay_writing",
        "summarization",
        "math_reasoning",
        "ethical_reasoning",
    ], (
        f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
    )

    reward_model_address = training_args.reward_model_address
    if reward_model_address is None:
        raise ValueError("Reward model address is not provided. Please provide a valid reward model address.")
    meta_reward_model_address = training_args.meta_reward_model_address
    if meta_reward_model_address is None:
        raise ValueError(
            "Meta reward model address is not provided. Please provide a valid meta reward model address."
        )

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    base_causal_model = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
    )
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=base_causal_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, torch_dtype=torch.bfloat16, trust_remote_code=model_args.trust_remote_code
    )

    # Override peft config for reproducibility
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        target_modules=[
            "down_proj",
            "gate_proj",
            "k_proj",
            "lm_head",
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        ],
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        use_rslora=False,
        use_dora=False,
    )
    ref_policy = None

    ################
    # Dataset
    ################
    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = get_task_dataset(task_name, tokenizer, "train")
        eval_dataset = get_task_dataset(task_name, tokenizer, "test")

    ################
    # Training
    ################
    trainer = MPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model_address=reward_model_address,
        meta_reward_model_address=meta_reward_model_address,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)

    # trainer.generate_completions()
