#!/usr/bin/env bash
# ------------------------------------------------------------------
# launch_rm_mrm_local.sh  RM_SUFFIX [RM_PORT]
#
# • GPUs 2 is reserved for the reward-model server.
# • GPUs 0-1 are left free for your accelerate training job.
# ------------------------------------------------------------------
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 <rm_suffix> [rm_port]"
  exit 1
fi

RM_SUFFIX=$1
RM_PORT=${2:-30000}

SGLANG_VENV="$HOME/Development/sglang/sglang-venv"
LOG_DIR="$HOME/Development/sglang/logs"
mkdir -p "$LOG_DIR"

########################################################################
# Helpers
########################################################################
kill_gpu_procs() { sudo pkill -f sglang 2>/dev/null || true; }

start_server() {
  local gpus=$1 dp=$2 suffix=$3 port=$4 log=$5
  echo "[INFO] Starting Qwen/Qwen2.5-${suffix^^}-Instruct-AWQ "\
       "on GPU(s) $gpus (port $port)"
  tmux new-session -d -s "sg_${suffix}_${port}" bash -c '
    set -euo pipefail
    source "'"$SGLANG_VENV"'/bin/activate"
    CUDA_VISIBLE_DEVICES='"$gpus"' \
    exec python -m sglang.launch_server \
      --model-path "Qwen/Qwen2.5-'"${suffix^^}"'-Instruct-AWQ" \
      --host 0.0.0.0 --port '"$port"' \
      --mem-fraction-static 0.9 --dp '"$dp"' \
      --schedule-conservativeness 0.35 \
      --attention-backend flashinfer \
      --enable-torch-compile \
      --grammar-backend outlines \
      &> "'"$log"'"
  '
}

wait_ready() {
  local port=$1 tag=$2
  echo -n "[INFO] Waiting for $tag on port $port "
  for _ in {1..60}; do
    if (exec 3<>"/dev/tcp/0.0.0.0/$port") 2>/dev/null; then
      exec 3<&- 3>&-
      echo "up!!!"; return 0
    fi
    printf '.'; sleep 5
  done
  echo "✗ timeout — check $LOG_DIR/${tag}.log"; exit 1
}

########################################################################
# Main logic
########################################################################
echo "[INFO] Cleaning up any old sglang jobs…"; kill_gpu_procs; sleep 5

start_server "2"   1 "$RM_SUFFIX"  "$RM_PORT"  "$LOG_DIR/rm.log"
wait_ready  "$RM_PORT"  "rm"

echo "[INFO] Reward-model server is running."