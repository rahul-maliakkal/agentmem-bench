#!/usr/bin/env bash
# Runs every pending experiment back to back. Start inside tmux:
#   tmux new -s pending ; bash run_pending.sh
# Each step skips work that already has results, so it is safe to restart.
set -uo pipefail
cd "$(dirname "$0")"
echo "### 1/4 Agents per GPU: finish Granite runs (Exp 7)"
SUFFIX=20261022_0208 bash run_concurrency.sh
echo "### 2/4 Repeated work: 1B pair and Devstral (Exp 4)"
bash run_multiturn.sh
echo "### 3/4 SGLang (Exp 6)"
bash run_sglang.sh
echo "### 4/4 Agents per GPU: Devstral vs Nemotron (Exp 7, large pair)"
SUFFIX=large LARGE=1 ONLY="devstral24b_dense nemo3_nano30b" bash run_concurrency.sh
tar czf results_pending.tar.gz results
echo "ALL DONE. Download results_pending.tar.gz"
