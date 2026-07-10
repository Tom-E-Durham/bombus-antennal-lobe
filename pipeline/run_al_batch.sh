#!/bin/bash
# Batch AL segmentation (pipeline stage A).
# Finds every AL*.tif that has a curated processed_binary_mask and runs 02_segment_al.py on it.
#
# Usage:
#   ./run_al_batch.sh [DATA_DIR]
# DATA_DIR defaults to `paths.data_root` in ../config.yaml and may be the whole confocal root
# (all species) or a single <species>/ or <sample>/ subdirectory.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Resolve DATA_DIR: CLI arg > config.yaml
DATA_DIR="${1:-$(python -c "import yaml; print(yaml.safe_load(open('$REPO_ROOT/config.yaml'))['paths']['data_root'])")}"

PROCESS_SCRIPT="$SCRIPT_DIR/02_segment_al.py"
LOG_DIR="$REPO_ROOT/results/batch_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_process_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/batch_errors_$(date +%Y%m%d_%H%M%S).log"

total_files=0; processed_files=0; skipped_files=0; error_files=0

echo "=== Batch AL segmentation started at $(date) ===" | tee -a "$MAIN_LOG"
echo "Data directory: $DATA_DIR" | tee -a "$MAIN_LOG"
echo "Process script: $PROCESS_SCRIPT" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Decide whether a given AL*.tif should be segmented.
should_process_file() {
    local tif_file="$1"
    local dir_path; dir_path=$(dirname "$tif_file")
    local filename; filename=$(basename "$tif_file" .tif)

    # Sample B284 is excluded from the study (segmentation artefact).
    [[ "$tif_file" == *"/B284"* ]] && return 1

    # Only AL1/AL2 stacks (skip whole-brain).
    [[ "${filename,,}" == *"wb"* || "${filename,,}" == *"wholebrain"* ]] && return 1
    [[ "${filename,,}" != *"al1"* && "${filename,,}" != *"al2"* ]] && return 1

    # Skip if already segmented.
    [[ -f "$dir_path/${filename}_seg_counts.csv" ]] && return 1

    # Pick the cache folder for this AL.
    local cache_folder
    if [[ "${filename,,}" == *"al1"* ]]; then cache_folder="cache_AL1"
    elif [[ "${filename,,}" == *"al2"* ]]; then cache_folder="cache_AL2"
    else return 1; fi

    # Require a curated binary mask in the cache.
    local n_masks; n_masks=$(find "$dir_path/$cache_folder" -name "processed_binary_mask_*.tif" -type f 2>/dev/null | wc -l)
    [[ "$n_masks" -gt 0 ]] && return 0 || return 1
}

process_single_file() {
    local tif_file="$1"
    echo "Processing: ${tif_file#$DATA_DIR/}" | tee -a "$MAIN_LOG"
    python -u "$PROCESS_SCRIPT" "$tif_file" 2>&1 | stdbuf -oL -eL tee -a "$MAIN_LOG"
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        echo "OK: ${tif_file#$DATA_DIR/}" | tee -a "$MAIN_LOG"; ((processed_files++))
    else
        echo "ERROR: ${tif_file#$DATA_DIR/}" | tee -a "$MAIN_LOG" "$ERROR_LOG"; ((error_files++))
    fi
    echo "" | tee -a "$MAIN_LOG"
}

TEMP_FILE_LIST=$(mktemp)
find "$DATA_DIR" -name "*.tif" -type f > "$TEMP_FILE_LIST"

while IFS= read -r tif_file; do
    ((total_files++))
    if should_process_file "$tif_file"; then
        process_single_file "$tif_file"
    else
        echo "Skipping: ${tif_file#$DATA_DIR/}" | tee -a "$MAIN_LOG"
        ((skipped_files++))
    fi
done < "$TEMP_FILE_LIST"
rm -f "$TEMP_FILE_LIST"

echo "=== Completed at $(date) ===" | tee -a "$MAIN_LOG"
echo "Total: $total_files | processed: $processed_files | skipped: $skipped_files | errors: $error_files" | tee -a "$MAIN_LOG"
[[ $error_files -gt 0 ]] && exit 1 || exit 0
