#!/usr/bin/env bash
# Launch E3 (style x decoding sweep) and E4 (alias-neuron find + steering sweep)
# in parallel. Both are detached, so the apps keep running on Modal if this
# shell or the laptop goes away. Each client streams its logs to its own file.
#
#   bash modal/run_e3_e4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

modal run --detach modal/e3_style_and_sampling.py::sweep  > logs/e3.log 2>&1 &
modal run --detach modal/e4_alias_neurons.py::pipeline    > logs/e4.log 2>&1 &
wait
