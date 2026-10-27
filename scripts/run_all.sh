#!/usr/bin/env bash
# One-command reproduction. Requires the fin-env environment and data/raw/ populated.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/fetch_data.py
python scripts/train_baseline.py
python scripts/run_m1.py
python scripts/run_m2.py
python scripts/run_m3.py
python scripts/run_m4.py
python scripts/run_m6.py
python scripts/run_m8.py
python scripts/run_policy.py
