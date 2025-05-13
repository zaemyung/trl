#!/bin/bash

trl_dir="/home/elicer/Development/trl"
SCRIPT="$trl_dir/examples/scripts/mpo.py"
WANDB_ENTITY="iterater"
WANDB_PROJECT="mpo-new"
DATASET="essay_writing"
TASK="essay_writing"
PROMPT_DIR="$trl_dir/trl/extras/mpo/prompts/essay_writing"


run_experiment() {
    local exp_type=$1
    local rubric_type=$2
    local rm=$3
    local mrm=$4
    local prompt_name=$5

    # Conditionally set CUDA_DEVICES
    local CUDA_DEVICES
    if [ "$rm" == "$mrm" ]; then
        CUDA_DEVICES="0,1,2,3"
    else
        CUDA_DEVICES="0,1"
    fi

    local model_name
    if [ "$exp_type" == "mpo" ]; then
        model_name="${rubric_type}-${rm}_${mrm}"
    else
        model_name="${rubric_type}-${rm}"
    fi

    # Count GPUs based on CUDA_DEVICES
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_DEVICES"
    local NUM_GPUS=${#GPU_ARRAY[@]}
    local ACC_CONFIG="$trl_dir/examples/accelerate_configs/deepspeed_zero2_${NUM_GPUS}gpus.yaml"

    local policy_model="policy-3b"
    local param_size="${mrm^^}"
    local exp_name="${policy_model}-ew-${exp_type}-${model_name}"
    local output_dir="$trl_dir/models/${policy_model}/${TASK}/${exp_type}/${model_name}"
    local grad_acc_steps=4
    if [ "$NUM_GPUS" -eq 2 ]; then
        grad_acc_steps=8
    fi
    local num_mpo_interval=99999999
    if [ "$exp_type" == "mpo" ]; then
        num_mpo_interval=10
    fi

    local rm_address="http://ymscfzegfxjbrdnk.tunnel.elice.io"
    local mrm_address=${rm_address}
    local sglang_pid=""
    if [ "$exp_type" == "mpo" ] && [ "$rm" != "$mrm" ]; then
        mrm_address="http://0.0.0.0:30000"
        echo "Starting sglang.launch_server in background..."
        CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
            --model-path Qwen/Qwen2.5-"$param_size"-Instruct-AWQ \
            --host 0.0.0.0 --port 30000 \
            --mem-fraction-static 0.9 --dp 2 \
            --schedule-conservativeness 0.35 \
            --attention-backend flashinfer \
            --enable-torch-compile > mrm_sglang_server.log 2>&1 &

        sglang_pid=$!
        echo "sglang PID: $sglang_pid"

        echo "Waiting for sglang server to be ready..."
        until curl -s "$mrm_address" > /dev/null; do
            echo "Waiting for server at $mrm_address..."
            sleep 3
        done
        echo "sglang server is ready."
    fi

    printf "Running: $exp_name with $NUM_GPUS GPUs and ACC_CONFIG: $ACC_CONFIG\n"

    printf "========== Experiment Configuration ==========\n"
    printf "CUDA_DEVICES       : $CUDA_DEVICES\n"
    printf "NUM_GPUS           : $NUM_GPUS\n"
    printf "exp_type           : $exp_type\n"
    printf "rubric_type        : $rubric_type\n"
    printf "rm                 : $rm\n"
    printf "mrm                : $mrm\n"
    printf "param_size         : $param_size\n"
    printf "prompt_name        : $prompt_name\n"
    printf "policy_model       : $policy_model\n"
    printf "exp_name           : $exp_name\n"
    printf "output_dir         : $output_dir\n"
    printf "grad_acc_steps     : $grad_acc_steps\n"
    printf "num_mpo_interval   : $num_mpo_interval\n"
    printf "rm_address         : $rm_address\n"
    printf "mrm_address        : $mrm_address\n"
    printf "==============================================\n\n"


    # CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" accelerate launch --config_file "$ACC_CONFIG" \
    #     "$SCRIPT" \
    #     --dataset_name "$DATASET" \
    #     --task_name "$TASK" \
    #     --wandb_entity "$WANDB_ENTITY" \
    #     --wandb_project "$WANDB_PROJECT" \
    #     --exp_name "$exp_name" \
    #     --init_rm_prompt "$PROMPT_DIR/$prompt_name" \
    #     --output_dir "$output_dir" \
    #     --learning_rate 3e-6 \
    #     --num_ppo_epochs 4 \
    #     --num_mpo_interval "$num_mpo_interval" \
    #     --save_n_updates 20 \
    #     --num_mpo_samples 20 \
    #     --num_mini_batches 1 \
    #     --per_device_train_batch_size 4 \
    #     --gradient_accumulation_steps "$grad_acc_steps" \
    #     --local_rollout_forward_batch_size 48 \
    #     --total_episodes 26013 \
    #     --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    #     --sft_model_path "Qwen/Qwen2.5-3B-Instruct" \
    #     --response_length 400 \
    #     --missing_eos_penalty 1.0 \
    #     --kl_coef 0.02 \
    #     --stop_token "eos" \
    #     --reward_model_address "$rm_address" \
    #     --meta_reward_model_address "$mrm_address"



    # sleep 10
}


# Run
exp_type="mpo"
rubric_type="iter0"
rm="1.5b"
# declare -a mrms=("1.5b" "3b" "7b" "14b")
declare -a mrms=("3b")
declare -a prompts=("evaluation_rubric_real_iter_0.txt")
for mrm in "${mrms[@]}"; do
    for prompt in "${prompts[@]}"; do
        run_experiment "$exp_type" "$rubric_type" "$rm" "$mrm" "$prompt"
    done
done