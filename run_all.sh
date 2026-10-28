#!/usr/bin/env bash
# Full matched three-head suite with live console + log file.
#
# Usage (from this directory on the VM):
#   chmod +x run_all.sh
#   ./run_all.sh --workers 16 --seeds 20260924,20260925,20260926
#   ./run_all.sh --workers 16 --seeds 20260924,20260925,20260926 --full
#
# Logs: logs/run_YYYYMMDD_HHMMSS.log (gitignored).
# Results: results/<dataset>/seed_<seed>/ and results/<dataset>_full/seed_<seed>/
# Multi-seed also writes results/<dataset>[_full]/aggregate.json

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WORKERS=8
DO_FULL=0
SKIP_FULL=0
DEVICE="cpu"
SEEDS="20260924,20260925,20260926"
PYTHON="${PYTHON:-python3}"
export TQDM_DISABLE=1

usage() {
  cat <<EOF
Usage: $0 [options]

  --workers N       DataLoader workers (default: ${WORKERS})
  --device NAME     cpu|cuda (default: ${DEVICE})
  --seeds LIST      comma-separated seeds (default: ${SEEDS})
  --seed N          single seed (overrides --seeds)
  --full            also run cifar10/omniglot --full after quick passes
  --skip-full       never run --full
  --python PATH     Python interpreter (default: python3, or \$PYTHON)
  -h, --help        show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed) SEEDS="$2"; shift 2 ;;
    --full) DO_FULL=1; shift ;;
    --skip-full) SKIP_FULL=1; DO_FULL=0; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${SKIP_FULL}" -eq 1 ]]; then
  DO_FULL=0
fi

mkdir -p logs
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="logs/run_${STAMP}.log"
FAIL=0

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    echo "No python interpreter found (tried ${PYTHON} and python)." >&2
    exit 127
  fi
fi

log() {
  local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$msg" | tee -a "$LOG"
}

run_step() {
  local title="$1"
  shift
  log "=== START: ${title} ==="
  log "CMD: $*"
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "=== FAIL (${rc}): ${title} ==="
    FAIL=1
    return "${rc}"
  fi
  log "=== DONE: ${title} ==="
  return 0
}

set -e

log "repo=${ROOT}"
log "python=$("${PYTHON}" -c 'import sys; print(sys.executable)')"
log "torch=$("${PYTHON}" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo missing)"
log "workers=${WORKERS} device=${DEVICE} seeds=${SEEDS} full=${DO_FULL}"
log "log_file=${ROOT}/${LOG}"

run_step "smoke_test" "${PYTHON}" smoke_test.py || exit 1

run_step "synthetic" \
  "${PYTHON}" run.py --dataset synthetic --device "${DEVICE}" --seeds "${SEEDS}" --workers "${WORKERS}" \
  || true

run_step "cifar10" \
  "${PYTHON}" run.py --dataset cifar10 --device "${DEVICE}" --seeds "${SEEDS}" --workers "${WORKERS}" \
  || true

run_step "omniglot" \
  "${PYTHON}" run.py --dataset omniglot --device "${DEVICE}" --seeds "${SEEDS}" --workers "${WORKERS}" \
  || true

if [[ "${DO_FULL}" -eq 1 ]]; then
  run_step "cifar10 --full" \
    "${PYTHON}" run.py --dataset cifar10 --full --device "${DEVICE}" --seeds "${SEEDS}" --workers "${WORKERS}" \
    || true
  run_step "omniglot --full" \
    "${PYTHON}" run.py --dataset omniglot --full --device "${DEVICE}" --seeds "${SEEDS}" --workers "${WORKERS}" \
    || true
fi

log "=== SUMMARY ==="
if [[ -d results ]]; then
  find results -type f \( -name 'metrics.json' -o -name 'aggregate.json' -o -name 'summary.json' -o -name '*.png' \) \
    | sort | tee -a "$LOG"
fi

# Print aggregate hold rates if present.
if compgen -G "results/*/aggregate.json" > /dev/null || compgen -G "results/*_full/aggregate.json" > /dev/null; then
  log "=== AGGREGATE HOLD RATES ==="
  "${PYTHON}" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
for p in sorted(Path("results").glob("**/aggregate.json")):
    a = json.loads(p.read_text())
    print(
        f"{p}: P1_hold={a.get('p1_hold_rate')} "
        f"P0_hold={a.get('p0_hold_rate')} "
        f"P2_hold_interp={a.get('p2_hold_rate')} "
        f"P2_status={a.get('p2_status_counts')} "
        f"warnings={a.get('warnings_any')}"
    )
PY
fi

if [[ "${FAIL}" -ne 0 ]]; then
  log "FINISHED WITH FAILURES — see ${LOG}"
  exit 1
fi

log "FINISHED OK — log at ${LOG}"
exit 0
