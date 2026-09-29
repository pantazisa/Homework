#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# run_hybrid_benchmark.sh
# ---------------------------------------------------------------------------
# Benchmarks crack_astar_hybrid (Pthreads + CUDA) by sweeping:
#   1. Thread count (fixed batch size)
#   2. GPU batch size (fixed thread count)
# Evaluates both length 4 and exploratory length 5 targets.
#
# USAGE:
#   scripts/run_hybrid_benchmark.sh [output_dir]
# ---------------------------------------------------------------------------

# Run from the project root regardless of where the script is invoked from,
# so the Makefile, sources in src/, and built binaries resolve correctly.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUT_DIR="${1:-reports_hybrid}"
mkdir -p "$OUT_DIR"
BIN="./crack_astar_hybrid"

# --- test cases -------------------------------------------------------------
# Length 4: identical targets to run_all_astar.sh's TEST_CASES, so results
# are directly comparable against every other implementation's numbers.
LENGTH4_CASES=(
    "random_a:4:PtYg"
    "random_b:4:j&mU"
    "random_c:4:=h#B"
    "random_d:4:el31"
    "random_e:4:iEl("
    "random_f:4:2h-p"
)

# Length 5: NEW, genuinely random, NOT pre-verified -- this is the actual
# experiment. Every other implementation failed 100% of the time (0/3
# tested) at this length without training.
LENGTH5_CASES=(
    "len5_a:5:yJuq)"
    "len5_b:5:ntG0y"
    "len5_c:5:(K5cq"
    "len5_d:5:=e!4f"
)

THREAD_COUNTS=(1 2 4 8 16 32)
BATCH_SIZES=(16384 65536 262144 1048576)

# Fixed values used for the sweep that ISN'T varying in a given phase.
FIXED_BATCH_SIZE=65536
FIXED_THREADS=16   # a reasonable middle value; adjust based on what the
                    # thread sweep phase shows as a good operating point

echo "Building hybrid implementation..."
if [ -f Makefile ] && grep -q "crack_astar_hybrid" Makefile 2>/dev/null; then
    make crack_astar_hybrid
elif [ -f src/crack_astar_hybrid.cu ]; then
    nvcc -O3 -arch=sm_70 -Isrc -o crack_astar_hybrid src/crack_astar_hybrid.cu src/astar_heap.c src/md5.c -lpthread 2>&1
else
    echo "src/crack_astar_hybrid.cu not found."
fi

if [ ! -x "$BIN" ]; then
    echo "WARNING: $BIN not built (no CUDA toolchain, or build failed above)."
    echo "This script will still run but every row will show 'not built'."
fi

{
    echo "============================================================"
    echo " Hybrid (Pthreads + CUDA) A* Benchmark Report"
    echo "============================================================"
    echo "Generated : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Host      : $(uname -srmo 2>/dev/null || uname -a)"
    echo "CPU cores : $(nproc 2>/dev/null || echo unknown)"
    echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
    echo "Note      : Benchmarks hybrid scaling on length 4 and length 5 targets."
    echo "============================================================"
} > "$OUT_DIR/hybrid_report.txt"

run_once() {
    local length="$1" target="$2" threads="$3" batch_size="$4"
    local output

    if [ ! -x "$BIN" ]; then
        echo "ERROR"
        return
    fi

    if ! output=$("$BIN" "$length" "$target" --threads "$threads" --batch-size "$batch_size" 2>&1); then
        echo "ERROR"
        return
    fi

    local elapsed nodes gpu_batches found

    elapsed=$(echo "$output" | grep -oP 'Elapsed time\s*:\s*\K[0-9.]+' | tail -1)
    nodes=$(echo "$output" | grep -oP 'Total nodes expanded \(all threads\)\s*:\s*\K[0-9]+')
    gpu_batches=$(echo "$output" | grep -oP 'Total GPU batches \(all threads/streams\)\s*:\s*\K[0-9]+')
    [ -z "$nodes" ] && nodes="-"
    [ -z "$gpu_batches" ] && gpu_batches="-"

    if echo "$output" | grep -q "^CRACKED"; then
        found="yes"
    else
        found="no"
    fi

    if [ -z "$elapsed" ]; then
        echo "ERROR"
    else
        echo "${elapsed}|${nodes}|${gpu_batches}|${found}"
    fi
}

run_case_table() {
    local case_list_name="$1"
    local -n cases_ref="$case_list_name"

    for case_def in "${cases_ref[@]}"; do
        IFS=':' read -r label length target <<< "$case_def"

        # --- Phase 1: thread sweep at fixed batch size ---
        echo
        echo "=== [$label] Thread sweep (length=$length, target=$target, batch-size=$FIXED_BATCH_SIZE) ==="
        {
            echo
            echo "------------------------------------------------------------"
            echo " [$label] THREAD SWEEP  (length=$length, target=$target, batch-size=$FIXED_BATCH_SIZE)"
            echo "------------------------------------------------------------"
            printf "%-10s %-12s %-14s %-12s %-8s\n" "Threads" "Time(s)" "NodesExpanded" "GPUBatches" "Found?"
            printf '%s\n' "------------------------------------------------------------------"
        } >> "$OUT_DIR/hybrid_report.txt"

        for t in "${THREAD_COUNTS[@]}"; do
            echo -n "  threads=$t... "
            result=$(run_once "$length" "$target" "$t" "$FIXED_BATCH_SIZE")
            if [ "$result" == "ERROR" ]; then
                echo "ERROR/not built"
                printf "%-10s %-12s %-14s %-12s %-8s\n" "$t" "ERROR" "-" "-" "-" >> "$OUT_DIR/hybrid_report.txt"
                continue
            fi
            elapsed="${result%%|*}"; rest="${result#*|}"
            nodes="${rest%%|*}"; rest="${rest#*|}"
            gpu_batches="${rest%%|*}"; found="${rest#*|}"
            echo "${elapsed}s (found: $found)"
            printf "%-10s %-12s %-14s %-12s %-8s\n" "$t" "$elapsed" "$nodes" "$gpu_batches" "$found" >> "$OUT_DIR/hybrid_report.txt"
        done

        # --- Phase 2: batch-size sweep at fixed thread count ---
        echo
        echo "=== [$label] Batch-size sweep (length=$length, target=$target, threads=$FIXED_THREADS) ==="
        {
            echo
            echo "------------------------------------------------------------"
            echo " [$label] BATCH-SIZE SWEEP  (length=$length, target=$target, threads=$FIXED_THREADS)"
            echo "------------------------------------------------------------"
            printf "%-12s %-12s %-14s %-12s %-8s\n" "BatchSize" "Time(s)" "NodesExpanded" "GPUBatches" "Found?"
            printf '%s\n' "------------------------------------------------------------------"
        } >> "$OUT_DIR/hybrid_report.txt"

        for b in "${BATCH_SIZES[@]}"; do
            echo -n "  batch-size=$b... "
            result=$(run_once "$length" "$target" "$FIXED_THREADS" "$b")
            if [ "$result" == "ERROR" ]; then
                echo "ERROR/not built"
                printf "%-12s %-12s %-14s %-12s %-8s\n" "$b" "ERROR" "-" "-" "-" >> "$OUT_DIR/hybrid_report.txt"
                continue
            fi
            elapsed="${result%%|*}"; rest="${result#*|}"
            nodes="${rest%%|*}"; rest="${rest#*|}"
            gpu_batches="${rest%%|*}"; found="${rest#*|}"
            echo "${elapsed}s (found: $found)"
            printf "%-12s %-12s %-14s %-12s %-8s\n" "$b" "$elapsed" "$nodes" "$gpu_batches" "$found" >> "$OUT_DIR/hybrid_report.txt"
        done
    done
}

echo "############################################################"
echo "# LENGTH 4 (comparison against other implementations)"
echo "############################################################"
{
    echo
    echo "############################################################"
    echo "# LENGTH 4 -- directly comparable to run_all_astar.sh results"
    echo "############################################################"
} >> "$OUT_DIR/hybrid_report.txt"
run_case_table LENGTH4_CASES

echo
echo "############################################################"
echo "# LENGTH 5 (EXPLORATORY -- everything else failed 100% here)"
echo "############################################################"
{
    echo
    echo "############################################################"
    echo "# LENGTH 5 -- Scaling test on larger search spaces"
    echo "############################################################"
} >> "$OUT_DIR/hybrid_report.txt"
run_case_table LENGTH5_CASES

{
    echo
    echo "============================================================"
    echo "Notes:"
    echo "  - 'GPUBatches' is the total number of GPU kernel launches"
    echo "    summed across all threads' independent streams."
    echo "  - FIXED_THREADS=$FIXED_THREADS was used for the batch-size sweep."
    echo "============================================================"
} >> "$OUT_DIR/hybrid_report.txt"

echo
echo "Report written to: $OUT_DIR/hybrid_report.txt"
echo
cat "$OUT_DIR/hybrid_report.txt"