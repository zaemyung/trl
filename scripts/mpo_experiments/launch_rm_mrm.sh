#!/usr/bin/env bash
# ------------------------------------------------------------------
# launch_rm_mrm.sh PEM_KEY USER@HOST RM_SUFFIX MRM_SUFFIX [RM_PORT] [MRM_PORT]
# ------------------------------------------------------------------
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 <pem_key> <user@host> <rm_suffix> <mrm_suffix> [rm_port] [mrm_port]"
  exit 1
fi

PEM_KEY=$1
REMOTE_HOST=$2
RM_SUFFIX=$3
MRM_SUFFIX=$4
RM_PORT=32000
MRM_PORT=34000

ssh -i "$PEM_KEY" -p 24411 "$REMOTE_HOST" bash -s -- "$RM_SUFFIX" "$MRM_SUFFIX" "$RM_PORT" "$MRM_PORT" <<'REMOTE'
set -euo pipefail

RM_SUFFIX=$1
MRM_SUFFIX=$2
RM_PORT=$3
MRM_PORT=$4

SGLANG_VENV="$HOME/Development/sglang/sglang-venv"
LOG_DIR="$HOME/Development/sglang/logs"
mkdir -p "$LOG_DIR"

kill_gpu_procs () {
  # Ignore any GPU arguments; just make sure no lingering sglang servers remain.
  # -f  : match against the full command line
  # -9  : force-kill
  # 2>/dev/null prevents “no process found” noise
  sudo pkill -f sglang 2>/dev/null || true
}

echo "[INFO] Cleaning GPUs"
kill_gpu_procs
sleep 5

start_server () {
  local gpus=$1
  local dp=$2
  local suffix=$3
  local port=$4
  local log=$5

  echo "[INFO] Starting Qwen/Qwen2.5-${suffix^^}-Instruct-AWQ on GPUs $gpus (port $port)"
  # Run detached in its own tmux so the ssh session can close.
  tmux new-session -d -s "sg_${suffix}_${port}" "
    source \"$SGLANG_VENV/bin/activate\";
    CUDA_VISIBLE_DEVICES=$gpus \
    exec python -m sglang.launch_server \
      --model-path \"Qwen/Qwen2.5-${suffix^^}-Instruct-AWQ\" \
      --host 0.0.0.0 --port $port \
      --mem-fraction-static 0.9 --dp $dp \
      --schedule-conservativeness 0.35 \
      --attention-backend flashinfer \
      --enable-torch-compile \
      --grammar-backend outlines    &> \"$log\"
  "
}

start_server "0,1,2" 3 "$RM_SUFFIX"  "$RM_PORT"  "$LOG_DIR/rm.log"
start_server "3"      1 "$MRM_SUFFIX" "$MRM_PORT" "$LOG_DIR/mrm.log"

# ------------------------------------------------------------------
# Wait until each port answers; timeout after ~5 min to avoid hanging
# ------------------------------------------------------------------
wait_ready () {
  local port=$1          # e.g. 30001
  local tag=$2           # "rm" | "mrm" just for log messages

  echo -n "[INFO] Waiting for $tag on port $port "
  for _ in {1..60}; do          # 60 x 5 s ≈ 5 min max
    # Try to open a TCP connection; success means the server is up.
    if (exec 3<>"/dev/tcp/0.0.0.0/$port") 2>/dev/null; then
      exec 3<&- 3>&-              # close the temporary FD
      echo "✓ up"
      return 0
    fi
    printf '.'
    sleep 5
  done
  echo "✗ timeout — check the log for $tag"
  exit 1
}

wait_ready "$RM_PORT"  "rm"
wait_ready "$MRM_PORT" "mrm"
echo "[INFO] Both servers are running."
REMOTE