#!/bin/bash

trl_dir="/home/elicer/Development/trl"

########## MPO ##########
exp_type="mpo"
prompt_name="evaluation_rubric_real_iter_0.txt"
policy_model="policy-3b"

# policy 3b, MPO, 1.5b_1.5b
rm="1.5b"
mrm="1.5b"

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file $trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml \
    $trl_dir/examples/scripts/mpo.py \
    --dataset_name "essay_writing" \
    --task_name "essay_writing" \
    --wandb_entity "iterater" \
    --wandb_project "mpo-new" \
    --exp_name ${policy_model}-ew-${exp_type}-${rm}_${mrm} \
    --init_rm_prompt $trl_dir/trl/extras/mpo/prompts/essay_writing/${prompt_name} \
    --output_dir $trl_dir/models/${policy_model}/essay_writing/$exp_type/${rm}_${mrm} \
    --learning_rate 3e-6 \
    --num_ppo_epochs 4 \
    --num_mpo_interval 2 \
    --save_n_updates 20 \
    --num_mpo_samples 20 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --local_rollout_forward_batch_size 48 \
    --total_episodes 26013 \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --sft_model_path "Qwen/Qwen2.5-3B-Instruct" \
    --response_length 400 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.02 \
    --stop_token "eos" \
    --reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io" \
    --meta_reward_model_address "http://ymscfzegfxjbrdnk.tunnel.elice.io"
sleep 10

# policy 3b, MPO, 1.5b_3b
rm="1.5b"
mrm="3b"

# policy 3b, MPO, 1.5b_7b
mrm="3b"
mrm="7b"

# policy 3b, MPO, 1.5b_14b
mrm="3b"
mrm="14b"