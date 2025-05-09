#!/bin/bash

trl_dir="/home/elicer/Development/trl"

# # policy 1.5b, PPO 32b, autoprompt
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "essay_writing" \
#     --task_name "essay_writing" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "ew-ppo-autoprompt-32b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/essay_writing/evaluation_rubric_autoprompt.txt \
#     --output_dir $trl_dir/models/essay_writing/ppo/autoprompt-32b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 99999999 \
#     --num_mpo_samples 20 \
#     --save_n_updates 20 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --local_rollout_forward_batch_size 48 \
#     --total_episodes 26013 \
#     --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
#     --sft_model_path Qwen/Qwen2-1.5B-Instruct \
#     --response_length 400 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.02 \
#     --stop_token eos \
#     --reward_model_address "http://0.0.0.0:30000" \
#     --meta_reward_model_address "http://0.0.0.0:30000"

sleep 10

# policy 1.5b, PPO 72b, autoprompt
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml \
    $trl_dir/examples/scripts/mpo.py \
    --dataset_name "essay_writing" \
    --task_name "essay_writing" \
    --wandb_entity "iterater" \
    --wandb_project "mpo-new" \
    --exp_name "ew-ppo-autoprompt-72b" \
    --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/essay_writing/evaluation_rubric_autoprompt.txt \
    --output_dir $trl_dir/models/essay_writing/ppo/autoprompt-72b \
    --learning_rate 3e-6 \
    --num_ppo_epochs 4 \
    --num_mpo_interval 99999999 \
    --num_mpo_samples 20 \
    --save_n_updates 20 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --local_rollout_forward_batch_size 48 \
    --total_episodes 26013 \
    --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --sft_model_path Qwen/Qwen2-1.5B-Instruct \
    --response_length 400 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.02 \
    --stop_token eos \
    --reward_model_address "http://0.0.0.0:30000" \
    --meta_reward_model_address "http://0.0.0.0:30000"

sleep 10

# policy 1.5b, PPO 72b, expert
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml \
    $trl_dir/examples/scripts/mpo.py \
    --dataset_name "essay_writing" \
    --task_name "essay_writing" \
    --wandb_entity "iterater" \
    --wandb_project "mpo-new" \
    --exp_name "ew-ppo-expert-72b" \
    --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/essay_writing/evaluation_rubric_expert.txt \
    --output_dir $trl_dir/models/essay_writing/ppo/expert-72b \
    --learning_rate 3e-6 \
    --num_ppo_epochs 4 \
    --num_mpo_interval 99999999 \
    --num_mpo_samples 20 \
    --save_n_updates 20 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --local_rollout_forward_batch_size 48 \
    --total_episodes 26013 \
    --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --sft_model_path Qwen/Qwen2-1.5B-Instruct \
    --response_length 400 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.02 \
    --stop_token eos \
    --reward_model_address "http://0.0.0.0:30000" \
    --meta_reward_model_address "http://0.0.0.0:30000"