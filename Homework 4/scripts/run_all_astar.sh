#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# run_all_astar.sh
# ---------------------------------------------------------------------------
# Benchmarks the A* password cracker implementations (Sequential, Pthreads,
# OpenMP, OpenCilk, CUDA) across predefined test cases.
#
# USAGE:
#   scripts/run_all_astar.sh [output_dir]
# ---------------------------------------------------------------------------

# Run from the project root regardless of where the script is invoked from,
# so the Makefile, sources in src/, and built binaries resolve correctly.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUT_DIR="${1:-reports_astar}"
mkdir -p "$OUT_DIR"

# Predefined length-4 benchmark test cases
TEST_CASES=(
    "random_a:4:PtYg"      # ~0.85s
    "random_b:4:j&mU"      # ~2.74s
    "random_c:4:=h#B"      # ~5.40s
    "random_d:4:el31"      # ~7.17s
    "random_e:4:iEl("      # ~8.45s
    "random_f:4:2h-p"      # ~9.05s -- hardest confirmed case, best for showing speedup
)

# Thread counts and GPU batch sizes for benchmarks
THREAD_COUNTS=(1 2 4 8 16 32)
GPU_BATCH_SIZES=(65536 262144 1048576 4194304)

# --- implementations: "label:binary:mode" -----------------------------------
IMPLS=(
    "sequential:./crack_astar:seq"
    "pthreads:./crack_astar_pthreads:cpu"
    "openmp:./crack_astar_omp:cpu"
    "opencilk:./crack_astar_opencilk:cpu"
    "cuda:./crack_astar_cuda:gpu"
)

echo "Building CPU-based implementations (sequential, pthreads, OpenMP)..."
if [ -f Makefile ]; then
    make crack_astar crack_astar_pthreads crack_astar_omp
else
    gcc -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L -Isrc -o crack_astar src/crack_astar.c src/astar_heap.c src/md5.c -lm
    gcc -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L -pthread -Isrc -o crack_astar_pthreads src/crack_astar_pthreads.c src/astar_heap.c src/md5.c -lm
    gcc -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L -fopenmp -Isrc -o crack_astar_omp src/crack_astar_omp.c src/astar_heap.c src/md5.c -lm
fi

{
    echo "============================================================"
    echo " A* Search Benchmark Report"
    echo "============================================================"
    echo "Generated : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Host      : $(uname -srmo 2>/dev/null || uname -a)"
    echo "CPU cores : $(nproc 2>/dev/null || echo unknown)"
    echo "Note      : Uniform transition costs across the 76-character alphabet."
    echo "============================================================"
} > "$OUT_DIR/astar_report.txt"

run_once() {
    local bin="$1" length="$2" target="$3" threads="$4"
    local cmd output

    if [ "$threads" == "-" ]; then
        cmd=("$bin" "$length" "$target")
    else
        cmd=("$bin" "$length" "$target" --threads "$threads")
    fi

    if ! output=$("${cmd[@]}" 2>&1); then
        echo "ERROR"
        return
    fi

    local elapsed nodes found

    elapsed=$(echo "$output" | grep -oP 'Elapsed time\s*:\s*\K[0-9.]+' | tail -1)

    # "nodes expanded" wording differs: sequential/CUDA report a single
    # "A*:"/"Search:" line; CPU-parallel versions report a per-thread total.
    nodes=$(echo "$output" | grep -oP 'A\*: \K[0-9]+' | head -1)
    if [ -z "$nodes" ]; then
        nodes=$(echo "$output" | grep -oP 'Search: \K[0-9]+' | head -1)
    fi
    if [ -z "$nodes" ]; then
        nodes=$(echo "$output" | grep -oP 'Total nodes expanded \(all threads\)\s*:\s*\K[0-9]+')
    fi
    [ -z "$nodes" ] && nodes="-"

    if echo "$output" | grep -q "^CRACKED"; then
        found="yes"
    else
        found="no"
    fi

    if [ -z "$elapsed" ]; then
        echo "ERROR"
    else
        echo "${elapsed}|${nodes}|${found}"
    fi
}

for case_def in "${TEST_CASES[@]}"; do
    IFS=':' read -r label length target <<< "$case_def"

    echo
    echo "=== Test case: $label (length=$length, target=$target) ==="

    {
        echo
        echo "------------------------------------------------------------"
        echo " Test case: $label   (length=$length, target=$target)"
        echo "------------------------------------------------------------"
        printf "%-12s %-8s %-12s %-14s %-8s\n" \
            "Impl" "Threads" "Time(s)" "NodesExpanded" "Found?"
        printf '%s\n' "------------------------------------------------------------------"
    } >> "$OUT_DIR/astar_report.txt"

    for impl_def in "${IMPLS[@]}"; do
        IFS=':' read -r impl_label bin mode <<< "$impl_def"

        if [ ! -x "$bin" ]; then
            echo "  skipping $impl_label: $bin not found/built yet"
            printf "%-12s %-8s %-12s %-14s %-8s\n" \
                "$impl_label" "-" "not built" "-" "-" >> "$OUT_DIR/astar_report.txt"
            continue
        fi

        if [ "$mode" == "seq" ]; then
            echo -n "  running: $impl_label (length $length, target=$target)... "
            result=$(run_once "$bin" "$length" "$target" "-")
            if [ "$result" == "ERROR" ]; then
                echo "ERROR"
                printf "%-12s %-8s %-12s %-14s %-8s\n" \
                    "$impl_label" "1" "ERROR" "-" "-" >> "$OUT_DIR/astar_report.txt"
            else
                elapsed="${result%%|*}"; rest="${result#*|}"
                nodes="${rest%%|*}"; found="${rest#*|}"
                echo "${elapsed}s (found: $found)"
                printf "%-12s %-8s %-12s %-14s %-8s\n" \
                    "$impl_label" "1" "$elapsed" "$nodes" "$found" >> "$OUT_DIR/astar_report.txt"
            fi
            continue
        fi

        local_threads=("${THREAD_COUNTS[@]}")
        [ "$mode" == "gpu" ] && local_threads=("${GPU_BATCH_SIZES[@]}")

        for t in "${local_threads[@]}"; do
            echo -n "  running: $impl_label ($t threads, length $length, target=$target)... "
            result=$(run_once "$bin" "$length" "$target" "$t")
            if [ "$result" == "ERROR" ]; then
                echo "ERROR"
                printf "%-12s %-8s %-12s %-14s %-8s\n" \
                    "$impl_label" "$t" "ERROR" "-" "-" >> "$OUT_DIR/astar_report.txt"
                continue
            fi
            elapsed="${result%%|*}"; rest="${result#*|}"
            nodes="${rest%%|*}"; found="${rest#*|}"
            echo "${elapsed}s (found: $found)"
            printf "%-12s %-8s %-12s %-14s %-8s\n" \
                "$impl_label" "$t" "$elapsed" "$nodes" "$found" >> "$OUT_DIR/astar_report.txt"
        done
    done
done

{
    echo
    echo "============================================================"
    echo "Notes:"
    echo "  - 'Found? = no' indicates search budget or timeout reached."
    echo "  - For CUDA, 'Threads' indicates the candidate leaf batch size."
    echo "  - 'not built' indicates the binary was not compiled."
    echo "============================================================"
} >> "$OUT_DIR/astar_report.txt"

echo
echo "Report written to: $OUT_DIR/astar_report.txt"
echo
cat "$OUT_DIR/astar_report.txt"