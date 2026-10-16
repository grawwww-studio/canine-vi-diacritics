#!/usr/bin/env bash
# One command: setup (pinned) -> data -> train -> eval -> results.json / results.md
#   ./run.sh
# Optional: GPU_WRAP="<command that wraps GPU jobs>" ./run.sh   (e.g. a lock or scheduler on a shared GPU)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DISABLE_NATIVE_JIT=1   # torch 2.14 native Triton kernels need a C compiler; the stock path does not
mkdir -p work results
now() { date +%s.%N; }
el() { awk -v a="$(now)" -v b="$1" 'BEGIN{printf "%.2f", a-b}'; }   # elapsed seconds since $1
t_all=$(now)

# 1. setup: Python 3.12 venv + pinned requirements (needs `uv`: https://docs.astral.sh/uv/)
t=$(now)
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -q -r requirements.txt
t_setup=$(el $t)
PY=.venv/bin/python

# 2. data (CPU; skipped if work/*.jsonl already exist -- delete work/ to rebuild)
t=$(now)
[ -s work/test.jsonl ] || $PY scripts/diac.py prepare
t_data=$(el $t)

# 3. train (GPU)
t=$(now)
${GPU_WRAP:-} $PY scripts/diac.py train --bs 64 --epochs 3 --lr 5e-5
t_train=$(el $t)

# 4. eval on the held-out test set (GPU)
t=$(now)
${GPU_WRAP:-} $PY scripts/diac.py eval
t_eval=$(el $t)

t_total=$(el $t_all)
cat > results/timing.json <<JSON
{"setup_seconds": $t_setup, "data_seconds": $t_data, "train_seconds_wall_incl_load": $t_train,
 "eval_seconds_wall_incl_load": $t_eval, "total_seconds": $t_total}
JSON

# 5. results.json + results.md
$PY scripts/make_results.py
echo "done: results.md results.json"
