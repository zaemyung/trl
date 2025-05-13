#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#  Paths & constants
###############################################################################
trl_dir="/home/elicer/Development/trl"
SCRIPT="$trl_dir/examples/scripts/mpo.py"

WANDB_ENTITY="iterater"
WANDB_PROJECT="mpo-new"
DATASET="essay_writing"
TASK="essay_writing"
PROMPT_DIR="$trl_dir/trl/extras/mpo/prompts/essay_writing"

# venv that contains sglang + AWQ weights
SGLANG_VENV="/home/elicer/Development/sglang/sglang-venv"
SGLANG_PY="$SGLANG_VENV/bin/python"          # absolute interpreter path

###############################################################################
#  Main runner
###############################################################################
run_experiment() {
    local exp_type=$1         # mpo / ppo …
    local rubric_type=$2      # e.g. iter0
    local rm=$3               # reward-model size          (e.g. 1.5b)
    local mrm=$4              # meta-reward-model size     (e.g. 3b)
    local prompt_name=$5      # prompt file

    # ------------------------------------------------------------------------
    #  GPU layout
    # ------------------------------------------------------------------------
    local CUDA_DEVICES
    if [[ "$rm" == "$mrm" ]]; then
        CUDA_DEVICES="0,1,2,3"        # RM and MRM share a single 4-GPU server
    else
        CUDA_DEVICES="0,1"            # policy side gets 2 GPUs; MRM gets 2/3 later
    fi
    IFS=',' read -ra gpu_arr <<<"$CUDA_DEVICES"
    local NUM_GPUS=${#gpu_arr[@]}
    local ACC_CONFIG="$trl_dir/examples/accelerate_configs/deepspeed_zero2_${NUM_GPUS}gpus.yaml"

    # ------------------------------------------------------------------------
    #  Naming & bookkeeping
    # ------------------------------------------------------------------------
    local policy_model="policy-3b"
    local model_name
    if [[ "$exp_type" == "mpo" ]]; then
        model_name="${rubric_type}-${rm}_${mrm}"
    else
        model_name="${rubric_type}-${rm}"
    fi

    local exp_name="${policy_model}-ew-${exp_type}-${model_name}"
    local output_dir="$trl_dir/models/${policy_model}/${TASK}/${exp_type}/${model_name}"

    # gradient accumulation scaling
    local grad_acc_steps=4
    [[ "$NUM_GPUS" -eq 2 ]] && grad_acc_steps=8

    # MPO interval
    local num_mpo_interval=99999999
    [[ "$exp_type" == "mpo" ]] && num_mpo_interval=10

    # ------------------------------------------------------------------------
    #  RM / MRM addresses (and optional background MRM server)
    # ------------------------------------------------------------------------
    local rm_address="http://ymscfzegfxjbrdnk.tunnel.elice.io"
    local mrm_address="$rm_address"
    local sglang_pid=""

    if [[ "$exp_type" == "mpo" && "$rm" != "$mrm" ]]; then
        mrm_address="http://0.0.0.0:30000"
        echo "🔹 Starting dedicated MRM server in its own venv …"

        # Launch in a *sub-shell* that activates sglang-venv, then exec's python.
        (
            source "$SGLANG_VENV/bin/activate"
            CUDA_VISIBLE_DEVICES=2,3 \
            exec python -m sglang.launch_server \
                 --model-path "Qwen/Qwen2.5-${mrm^^}-Instruct-AWQ" \
                 --host 0.0.0.0 --port 30000 \
                 --mem-fraction-static 0.9 --dp 2 \
                 --schedule-conservativeness 0.35 \
                 --attention-backend flashinfer \
                 --enable-torch-compile
        ) > mrm_sglang_server.log 2>&1 &

        sglang_pid=$!
        echo "   ↳ MRM PID: $sglang_pid"

        echo -n "   ↳ Waiting for server to be ready "
        until curl -s "$mrm_address" >/dev/null; do
            echo -n "."
            sleep 10
        done
        echo " success!"
    fi

    # ------------------------------------------------------------------------
    #  Display run-time configuration
    # ------------------------------------------------------------------------
    printf -- "==============  Experiment %s  ==============\n" "$exp_name"
    printf "CUDA_DEVICES       : %s\n" "$CUDA_DEVICES"
    printf "NUM_GPUS           : %s\n" "$NUM_GPUS"
    printf "ACC_CONFIG         : %s\n" "$ACC_CONFIG"
    printf "grad_acc_steps     : %s\n" "$grad_acc_steps"
    printf "num_mpo_interval   : %s\n" "$num_mpo_interval"
    printf "rm_address         : %s\n" "$rm_address"
    printf "mrm_address        : %s\n" "$mrm_address"
    printf -- "==============================================\n\n"

    # ------------------------------------------------------------------------
    #  Launch policy fine-tuning
    # ------------------------------------------------------------------------
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    accelerate launch --config_file "$ACC_CONFIG" \
        "$SCRIPT" \
        --dataset_name "$DATASET" \
        --task_name "$TASK" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_project "$WANDB_PROJECT" \
        --exp_name "$exp_name" \
        --init_rm_prompt "$PROMPT_DIR/$prompt_name" \
        --output_dir "$output_dir" \
        --learning_rate 3e-6 \
        --num_ppo_epochs 4 \
        --num_mpo_interval "$num_mpo_interval" \
        --save_n_updates 20 \
        --num_mpo_samples 20 \
        --num_mini_batches 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps "$grad_acc_steps" \
        --local_rollout_forward_batch_size 48 \
        --total_episodes 26013 \
        --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
        --sft_model_path   "Qwen/Qwen2.5-3B-Instruct" \
        --response_length 400 \
        --missing_eos_penalty 1.0 \
        --kl_coef 0.02 \
        --stop_token "eos" \
        --reward_model_address "$rm_address"  \
        --meta_reward_model_address "$mrm_address"

    # ------------------------------------------------------------------------
    #  Clean-up background MRM server
    # ------------------------------------------------------------------------
    if [[ -n "$sglang_pid" ]]; then
        echo "🔹 Shutting down MRM server (PID $sglang_pid)…"
        sleep 5
        pkill -f sglang
        echo -n "Waiting for all sglang processes to exit"
        while pgrep -f sglang >/dev/null; do
            echo -n "."
            sleep 3
        done
        echo "   ↳ MRM server stopped."
    fi
    sleep 3
}

###############################################################################
#  Sweep
###############################################################################
exp_type="mpo"
rubric_type="iter0"
rm="1.5b"
declare -a mrms=("3b" "1.5b" "7b" "14b")
declare -a prompts=("evaluation_rubric_real_iter_0.txt")

for mrm in "${mrms[@]}"; do
  for prompt in "${prompts[@]}"; do
      run_experiment "$exp_type" "$rubric_type" "$rm" "$mrm" "$prompt"
  done
done