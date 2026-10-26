#!/usr/bin/env bash
# Launch experiments on Modal, in parallel. Name the ones to run; with no names,
# all of them run. All are detached, so the apps keep running on Modal if this
# shell or the laptop goes away. Each client streams its logs to logs/<name>.log.
#
#   bash modal/run_experiments.sh             # e3 e4 e5 reeval
#   bash modal/run_experiments.sh e5          # E5 only
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

target() {
  case "$1" in
    e3)     echo modal/e3_style_and_sampling.py::sweep ;;  # style x decoding sweep
    e4)     echo modal/e4_alias_neurons.py::pipeline ;;    # alias-neuron find + steering
    e5)     echo modal/e5_neuron_diff.py::pipeline ;;      # copy / remove parts of GRPO's delta
    reeval) echo modal/reeval_official.py::main ;;         # CPU rescore of stored predictions
    *)      echo "unknown experiment: $1" >&2; return 1 ;;
  esac
}

[ $# -eq 0 ] && set -- e3 e4 e5 reeval
for n in "$@"; do target "$n" > /dev/null; done

for n in "$@"; do
  modal run --detach "$(target "$n")" > "logs/$n.log" 2>&1 &
done
wait
