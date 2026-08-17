#!/bin/bash
# One-shot status of the experiment: sweep progress and current leaderboard.
cd "$(dirname "$0")/.."

DONE=$(ls results/runs/*.json 2>/dev/null | grep -vc baselines)
ACTIVE=$(pgrep -fc "scripts/train.py" 2>/dev/null || echo 0)
echo "=== sweep: ${DONE}/56 runs done | ${ACTIVE} training processes active ==="
tail -3 logs/sweep.log 2>/dev/null
echo

python - <<'PY'
import json, glob, numpy as np
from collections import defaultdict

runs = defaultdict(list)
for p in glob.glob('results/runs/*.json'):
    if 'baselines' in p:
        continue
    r = json.load(open(p))
    runs[(r['tag'], r['split_mode'])].append(r['test_metrics'])

def agg(v, k):
    a = np.array([x[k] for x in v])
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)

# non-deep baselines share the split, so show them in the same table
for p in glob.glob('results/runs/baselines_*.json'):
    rec = json.load(open(p))
    for name, m in rec['results'].items():
        if not name.endswith('_importance'):
            runs[(name, rec['split_mode'])].append(m)

if not runs:
    raise SystemExit('no runs yet')

full = runs.get(('phantom_full', 'group'), [])
base = agg(full, 'ap')[0] if full else None

print(f"{'config':28s} {'split':6s} {'n':>2s} {'AUC':>16s} {'AP':>16s} {'dAP':>8s} {'P@R95':>7s}")
print('-' * 92)
for k in sorted(runs, key=lambda k: -agg(runs[k], 'ap')[0]):
    v = runs[k]
    au, as_ = agg(v, 'auc')
    ap, ps = agg(v, 'ap')
    pr, _ = agg(v, 'precision_at_recall_95')
    d = f"{ap-base:+.4f}" if base is not None else "     -"
    print(f"{k[0]:28s} {k[1]:6s} {len(v):2d} {au:.4f}+-{as_:.4f} {ap:.4f}+-{ps:.4f} {d:>8s} {pr:7.4f}")
PY
