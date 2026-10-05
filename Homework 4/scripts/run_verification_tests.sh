#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# run_verification_tests.sh
# ---------------------------------------------------------------------------
# Automated functional verification test suite.
# Confirms that each built implementation correctly recovers reference targets.
# ---------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

TARGET="PtYg"
LEN=4

echo "============================================================"
echo " Running Automated Verification Tests (Target: $TARGET, Len: $LEN)"
echo "============================================================"

PASSED=0
TOTAL=0

verify_binary() {
    local name="$1"
    local cmd=("$@")
    cmd=("${cmd[@]:1}")

    TOTAL=$((TOTAL + 1))
    printf "%-25s ... " "$name"

    local executable="${cmd[0]}"
    if [ "$executable" = "env" ]; then
        executable="${cmd[2]}"
    fi

    if [ ! -x "$executable" ]; then
        echo "SKIPPED (binary not built: $executable)"
        return
    fi

    local output
    if ! output=$("${cmd[@]}" 2>&1); then
        echo "FAIL (execution error)"
        return
    fi

    if echo "$output" | grep -q "CRACKED\s*:\s*$TARGET"; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL (target not recovered)"
    fi
}

# 1. Sequential
verify_binary "Sequential" ./crack_astar $LEN "$TARGET"

# 2. Pthreads
verify_binary "Pthreads (8 threads)" ./crack_astar_pthreads $LEN "$TARGET" --threads 8

# 3. OpenMP
verify_binary "OpenMP (8 threads)" ./crack_astar_omp $LEN "$TARGET" --threads 8

# 4. OpenCilk
verify_binary "OpenCilk (8 workers)" env CILK_NWORKERS=8 ./crack_astar_opencilk $LEN "$TARGET"

# 5. CUDA
verify_binary "CUDA (batch 262144)" ./crack_astar_cuda $LEN "$TARGET" --threads 262144

# 6. Hybrid
verify_binary "Hybrid (8 th, 65536 b)" ./crack_astar_hybrid $LEN "$TARGET" --threads 8 --batch-size 65536

echo "============================================================"
echo " Verification Summary: $PASSED / $TOTAL tests passed."
echo "============================================================"
