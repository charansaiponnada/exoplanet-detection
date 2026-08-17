#!/bin/bash
# Wait for the training sweep to finish, then run the whole analysis chain.
set -u
cd "$(dirname "$0")/.."

while pgrep -f "run_sweep.py" > /dev/null; do
    sleep 60
done
echo "=== sweep finished at $(date -u +'%H:%M UTC') ==="
ls results/runs/*.json | grep -vc baselines | xargs -I{} echo "{} runs completed"

echo; echo "=== baselines (all seeds) ==="
for s in 0 1 2 3 4; do
    python scripts/baselines.py --seed "$s" 2>&1 | grep -E '"(auc|ap)"' | head -4
done

echo; echo "=== evaluate ==="
python scripts/evaluate.py

echo; echo "=== zero-shot TESS transfer ==="
python scripts/transfer_tess.py --tag phantom_derived_only

echo; echo "=== figures ==="
python scripts/figures.py

echo; echo "=== ANALYSIS COMPLETE at $(date -u +'%H:%M UTC') ==="
bash scripts/status.sh
