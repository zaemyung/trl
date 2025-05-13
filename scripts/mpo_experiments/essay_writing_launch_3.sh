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
    local CUDA_DEVICES="0,1,2,3"
    local ACC_CONFIG="$trl_dir/examples/accelerate_configs/deepspeed_zero2_4gpus.yaml"

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

    # MPO interval
    local num_mpo_interval=99999999
    [[ "$exp_type" == "mpo" ]] && num_mpo_interval=10

    # ------------------------------------------------------------------------
    #  RM / MRM addresses (and optional background MRM server)
    # ------------------------------------------------------------------------
    local rm_address="https://ymscfzegfxjbrdnk.tunnel.elice.io"
    local mrm_address="https://phzthbsnckqtdrmq.tunnel.elice.io"

    # ------------------------------------------------------------------------
    #  Display run-time configuration
    # ------------------------------------------------------------------------
    printf -- "==============  Experiment %s  ==============\n" "$exp_name"
    printf "CUDA_DEVICES       : %s\n" "$CUDA_DEVICES"
    printf "ACC_CONFIG         : %s\n" "$ACC_CONFIG"
    printf "grad_acc_steps     : %s\n" "$grad_acc_steps"
    printf "num_mpo_interval   : %s\n" "$num_mpo_interval"
    printf "rm_address         : %s\n" "$rm_address"
    printf "mrm_address        : %s\n" "$mrm_address"
    printf -- "==============================================\n\n"

    # ------------------------------------------------------------------------
    #  Spin-up RM & MRM servers and wait for them to be ready
    # ------------------------------------------------------------------------
    bash $trl_dir/scripts/mpo_experiments/launch_rm_mrm.bash \
        ~/.ssh/elice-cloud-ondemand-9cc8a02a-574e-4098-befa-f912334e75c9.pem \
        elicer@central-02.tcp.tunnel.elice.io \
        "${rm^^}" "${mrm^^}"

    # ------------------------------------------------------------------------
    # Launch policy fine-tuning
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