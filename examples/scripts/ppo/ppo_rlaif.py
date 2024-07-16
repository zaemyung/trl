import multiprocessing
import shutil

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
import torch

from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.extras.prompt_based_reward_model import Scorer
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead


"""
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_rlaif.py \
    --output_dir models/minimal/ppo_rlaif \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --total_episodes 26013 \
    --base_model Qwen/Qwen2-1.5B \
    --model_name_or_path Qwen/Qwen2-1.5B \
    --sft_model_path Qwen/Qwen2-1.5B-Instruct \
    --local_rollout_forward_batch_size 2 \
    --non_eos_penalty \
    --response_length 400 \
    --num_sample_generations 20 \
    --kl_coef 0.007 \
    --stop_token eos
"""


if __name__ == "__main__":
    parser = HfArgumentParser((PPOv2Config, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

    base_causal_model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=base_causal_model
    )

    reward_model = Scorer()

    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )

    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    ################
    # Dataset
    ################
    raw_datasets = load_dataset("zaemyung/writing_prompts_collection")
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(sample):
            input_ids = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "You will act as an English writer and compose either an essay or a story depending on the instruction given below.\n\nInstructions:\n"
                        + sample["prompt"]
                        + "\n\nYour Writing:\n",
                    }
                ],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    assert (
        train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    ), "The last token should not be an EOS token"

    ################
    # Training
    ################
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
