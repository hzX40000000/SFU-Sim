#!/usr/bin/env bash
# Run every hardware verification on one GPU and store the logs under build/results/<TAG>_<ARCH>/.
#
# usage: test/gpu/run_all.sh TAG [ARCH]
#   TAG   short device name used for the result folder, e.g. a100 / rtx4060 / pro5000
#   ARCH  sm_80 / sm_89 / sm_120 ...; default: compute capability of the selected GPU
# env  : CUDA_VISIBLE_DEVICES selects the GPU; STEPS limits steps (e.g. STEPS="minimal f16x2")
#
# Steps that already ended with their PASS line are skipped, so an interrupted run can be resumed.
set -uo pipefail

TAG=${1:?usage: test/gpu/run_all.sh TAG [ARCH]}
ARCH=${2:-}
if [ -z "$ARCH" ]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n1 | tr -d ' .')
  [ -n "$cap" ] || { echo "cannot detect compute capability; pass ARCH explicitly" >&2; exit 2; }
  ARCH=sm_$cap
fi
OUT=build/results/${TAG}_${ARCH}
mkdir -p "$OUT"

{
  echo "arch: $ARCH"
  echo "--- gpu"
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv || echo "nvidia-smi unavailable"
  echo "--- nvcc"
  nvcc --version
  echo "--- g++"
  g++ --version | head -n1
} > "$OUT/env.txt" 2>&1

# name | make target | PASS line
STEP_TABLE="
selftest|selftest|MODEL SELF TEST PASS
offline|offline|PASS
negative|gpu-negative|NEGATIVE CONTROL PASS
minimal|gpu-minimal|PASS
extended_smoke|gpu-extended-smoke|EXTENDED F32 SMOKE PASS
extended|gpu-extended|EXTENDED F32 PASS
f16x2_smoke|gpu-f16x2-smoke|F16X2 SMOKE PASS
f16x2|gpu-f16x2|F16X2 PASS
round_smoke|gpu-round-smoke|ROUND F32 SMOKE PASS
round|gpu-round|ROUND F32 PASS
"

summary=$OUT/summary.txt
: > "$summary.tmp"
idx=0
while IFS='|' read -r name target pass; do
  [ -n "$name" ] || continue
  idx=$((idx + 1))
  if [ -n "${STEPS:-}" ]; then
    case " $STEPS " in *" $name "*) ;; *) continue ;; esac
  fi
  log=$(printf '%s/%02d_%s.log' "$OUT" "$idx" "$name")
  if [ -f "$log" ] && [ "$(grep -v '^#' "$log" | tail -n1)" = "$pass" ]; then
    echo "skip $name (already PASS)"
  else
    echo "=== $name: make ARCH=$ARCH $target"
    start=$(date +%s)
    make --no-print-directory ARCH="$ARCH" "$target" 2>&1 | tee "$log"
    echo "# elapsed_s $(( $(date +%s) - start ))" >> "$log"
  fi
  last=$(grep -v '^#' "$log" | tail -n1)
  status=FAIL; [ "$last" = "$pass" ] && status=PASS
  printf '%-15s %-4s %s\n' "$name" "$status" "$last" >> "$summary.tmp"
done <<< "$STEP_TABLE"

mv "$summary.tmp" "$summary"
echo "--- $summary"
cat "$summary"
! grep -q ' FAIL ' "$summary"
