#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

DRY_RUN=0
QUIET=0

usage() {
  cat <<'EOF'
Usage: cleanup-test-extended.sh [options]

Removes temporary/generated artifacts produced by mlir-test-files/test-extended.sh.

Options:
  -n, --dry-run   Show what would be removed without deleting
  -q, --quiet     Reduce output
  -h, --help      Show this help message
EOF
}

log() {
  if [[ "$QUIET" == "0" ]]; then
    echo "$*"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=1
      ;;
    -q|--quiet)
      QUIET=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

# Build a de-duplicated list of paths to remove.
declare -A REMOVE_SET=()

enqueue_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    REMOVE_SET["$path"]=1
  fi
}

# 1) Parallel worker roots from run_parallel_tests() (mktemp in SCRIPT_DIR).
while IFS= read -r path; do
  enqueue_path "$path"
done < <(find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d -name '.parallel_*' -print 2>/dev/null || true)

# 2) Global operation-count output roots.
if [[ -d "$DATA_DIR" ]]; then
  while IFS= read -r path; do
    enqueue_path "$path"
  done < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -name 'operation_counts_*' -print 2>/dev/null || true)

  # 3) Per-run JSON artifacts written to mlir-test-files/data.
  while IFS= read -r path; do
    enqueue_path "$path"
  done < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type f \
    \( -name '*_computed_*.json' -o -name '*_annotated_*.json' -o -name '*_execution_*.json' -o -name '*_openfhe-result_*.json' -o -name '*_lattigo-result_*.json' \) \
    -print 2>/dev/null || true)
fi

# 4) Per-test generated artifacts.
# A test directory is identified by the convention <name>/<name>.mlir.
while IFS= read -r test_dir; do
  test_name="$(basename "$test_dir")"

  # Generated IR artifacts.
  while IFS= read -r path; do
    enqueue_path "$path"
  done < <(find "$test_dir" -mindepth 1 -maxdepth 1 -type f \
    \( -name "${test_name}-middle.mlir" \
    -o -name "${test_name}-middle-params-*.mlir" \
    -o -name "${test_name}-bgv-*.mlir" \
    -o -name "${test_name}-openfhe-*.mlir" \
    -o -name "${test_name}-lattigo-*.mlir" \) \
    -print 2>/dev/null || true)

  # Generated OpenFHE code artifacts.
  while IFS= read -r path; do
    enqueue_path "$path"
  done < <(find "$test_dir" -mindepth 1 -maxdepth 1 -type f \
    \( -name "${test_name}_direct.h" -o -name "${test_name}_direct.cpp" \
    -o -name "${test_name}_closed.h" -o -name "${test_name}_closed.cpp" \
    -o -name "${test_name}_bisection.h" -o -name "${test_name}_bisection.cpp" \) \
    -print 2>/dev/null || true)

  # Generated Lattigo code artifacts.
  while IFS= read -r path; do
    enqueue_path "$path"
  done < <(find "$test_dir" -mindepth 1 -maxdepth 1 -type f \
    \( -name "${test_name}_*_lattigo.go" \
    -o -name "${test_name}_*_lattigo_run.go" \
    -o -name "${test_name}_*_lattigo_main.go" \) \
    -print 2>/dev/null || true)

  # Per-test logs produced by run_command.
  if [[ -d "$test_dir/logs" ]]; then
    enqueue_path "$test_dir/logs"
  fi

done < <(
  find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null \
    | while IFS= read -r candidate; do
        base="$(basename "$candidate")"
        if [[ -f "$candidate/$base.mlir" ]]; then
          printf '%s\n' "$candidate"
        fi
      done
)

if [[ ${#REMOVE_SET[@]} -eq 0 ]]; then
  log "No temporary artifacts found."
  exit 0
fi

mapfile -t REMOVE_PATHS < <(printf '%s\n' "${!REMOVE_SET[@]}" | sort)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: ${#REMOVE_PATHS[@]} paths would be removed"
  printf '  %s\n' "${REMOVE_PATHS[@]}"
  exit 0
fi

echo "Removing ${#REMOVE_PATHS[@]} paths..."
for path in "${REMOVE_PATHS[@]}"; do
  rm -rf -- "$path"
done

echo "Cleanup complete."
