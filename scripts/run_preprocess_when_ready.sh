#!/bin/bash
# Wait for the Kepler bulk download to finish, then build the view tensors.
set -u
cd "$(dirname "$0")/.."

while pgrep -f "download_kepler.py" > /dev/null; do
    sleep 60
done
echo "download finished at $(date)"
du -sh data/kepler
find data/kepler -name '*.fits' | wc -l

python scripts/preprocess.py --workers 36 --out data/processed/dr24_views.h5
echo "preprocessing finished at $(date)"
