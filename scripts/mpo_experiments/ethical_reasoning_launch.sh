#!/bin/bash

trl_dir="/home/elicer/Development/trl"

# # policy 1.5b, PPO 32b, autoprompt
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-ppo-autoprompt-32b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_autoprompt.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/ppo/autoprompt-32b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 99999999 \
#     --num_mpo_samples 20 \
#     --save_n_updates 20 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --local_rollout_forward_batch_size 48 \
#     --total_episodes 13000 \
#     --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
#     --sft_model_path Qwen/Qwen2-1.5B-Instruct \
#     --response_length 600 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.02 \
#     --stop_token eos \
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"

# # policy 1.5b, PPO 72b, autoprompt
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-ppo-autoprompt-72b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_autoprompt.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/ppo/autoprompt-72b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 99999999 \
#     --num_mpo_samples 20 \
#     --save_n_updates 20 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --local_rollout_forward_batch_size 48 \
#     --total_episodes 13000 \
#     --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
#     --sft_model_path Qwen/Qwen2-1.5B-Instruct \
#     --response_length 600 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.02 \
#     --stop_token eos \
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"

# sleep 10

# # policy 1.5b, PPO 72b, iter0
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-ppo-iter0-72b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/ppo/iter0-72b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 99999999 \
#     --num_mpo_samples 20 \
#     --save_n_updates 20 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --local_rollout_forward_batch_size 48 \
#     --total_episodes 13000 \
#     --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
#     --sft_model_path Qwen/Qwen2-1.5B-Instruct \
#     --response_length 600 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.02 \
#     --stop_token eos \
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"

# sleep 10

# # policy 1.5b, PPO 32b, iter0
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-ppo-iter0-32b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/ppo/iter0-32b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 99999999 \
#     --num_mpo_samples 20 \
#     --save_n_updates 20 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --local_rollout_forward_batch_size 48 \
#     --total_episodes 13000 \
#     --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
#     --sft_model_path Qwen/Qwen2-1.5B-Instruct \
#     --response_length 600 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.02 \
#     --stop_token eos \
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"

# policy 1.5b, MPO 32b_32b
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
    $trl_dir/examples/scripts/mpo.py \
    --dataset_name "ethical_reasoning" \
    --task_name "ethical_reasoning" \
    --wandb_entity "iterater" \
    --wandb_project "mpo-new" \
    --exp_name "er-mpo-32b_32b" \
    --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
    --output_dir $trl_dir/models/ethical_reasoning/mpo/32b_32b \
    --learning_rate 3e-6 \
    --num_ppo_epochs 4 \
    --num_mpo_interval 99999999 \
    --num_mpo_samples 20 \
    --save_n_updates 20 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --local_rollout_forward_batch_size 48 \
    --total_episodes 13000 \
    --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --sft_model_path Qwen/Qwen2-1.5B-Instruct \
    --response_length 600 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.02 \
    --stop_token eos \
    --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
    --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"

# # policy 1.5b, MPO 72b_72B
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-mpo-72b_72b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/mpo/72b_72b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 20 \
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

# sleep 10

# # policy 1.5b, MPO 32b_72B
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-mpo-32b_72b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/mpo/32b_72b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 20 \
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
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://0.0.0.0:30001"

# # policy 1.5b, MPO 72b_32B
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-mpo-72b_32b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/mpo/72b_32b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 20 \
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
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://0.0.0.0:30001"

# # policy 1.5b, MPO 32b_32B
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
#     $trl_dir/examples/scripts/mpo.py \
#     --dataset_name "ethical_reasoning" \
#     --task_name "ethical_reasoning" \
#     --wandb_entity "iterater" \
#     --wandb_project "mpo-new" \
#     --exp_name "er-mpo-32b_32b" \
#     --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/ethical_reasoning/evaluation_rubric_real_iter_0.txt \
#     --output_dir $trl_dir/models/ethical_reasoning/mpo/32b_32b \
#     --learning_rate 3e-6 \
#     --num_ppo_epochs 4 \
#     --num_mpo_interval 20 \
#     --num_mpo_samples 20 \
#     --save_n_updates 20 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --local_rollout_forward_batch_size 48 \
#     --total_episodes 26013 \
#     --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
#     --sft_model_path Qwen/Qwen2-1.5B-Instruct \
#     --response_length 400 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.02 \
#     --stop_token eos \
#     --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
#     --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"