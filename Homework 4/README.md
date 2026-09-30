# Parallel A* Password Recovery (Homework 4)

An MD5 password recovery tool formulated as a **memory-bounded A\* state-space search** over a prefix tree, implemented sequentially and parallelized across five paradigms:
- **Pthreads**: Dynamic 2-character prefix task queue via atomic operations.
- **OpenMP**: Dynamic task scheduling with `#pragma omp parallel for schedule(dynamic, 1)`.
- **OpenCilk**: Fine-grained task decomposition with work-stealing via `cilk_for`.
- **CUDA**: CPU-driven pointer graph traversal with GPU-accelerated batch leaf verification.
- **Hybrid (Pthreads + CUDA)**: Multi-threaded dynamic task distribution with per-thread independent CUDA streams, pinned host memory, and asynchronous memory transfers.

---

## 1. Algorithmic Formulation

In cryptographic hashing (MD5), the avalanche effect prevents measuring any continuous distance between an intermediate prefix and the target hash. Instead of brute-force guessing or trained heuristic models, the problem is formulated as a systematic, memory-bounded graph search over prefix trees:

* **State Space**: A candidate prefix string $s$ at tree depth $d$.
* **Alphabet**: Fixed 76-character set:
  ```
  abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+
  ```
* **Path Cost $g(n)$**: Accumulated depth:
  $$g(n) = d \times \text{EDGE\_COST} \quad (\text{with } \text{EDGE\_COST} = 1.0)$$
* **Heuristic $h(n)$**: Remaining depth estimate to target length $L$:
  $$h(n) = (L - d) \times \text{EDGE\_COST}$$
  *(Admissible and consistent, guaranteeing uniform frontier progression).*
* **Evaluation Function $f(n)$**:
  $$f(n) = g(n) + h(n) = L$$
  Since $f(n)$ is uniform across candidates of the target length, A* systematically expands the search frontier level by level.
* **Beam Bounding (Memory Management)**:
  To prevent combinatorial memory explosion:
  - The frontier is managed by a binary min-heap.
  - When the heap exceeds `BEAM_CAP = 3,000,000` states, an in-place `quickselect` trims the queue in expected $O(n)$ time, retaining only the best `BEAM_KEEP = 1,200,000` states before re-heapifying.
* **Search Budgets & Strict Evaluation**:
  - `TIME_BUDGET_SECONDS`: 60.0 seconds maximum wall-clock search time.
  - `NODE_BUDGET`: 30,000,000 nodes (35,000,000 in the Hybrid variant).
  - There is **no brute-force fallback**. If a target is not found within the search budget or beam retention, the program reports a genuine failure (`Password NOT recovered`).
* **Goal Test & MD5 Hashing**:
  - For testing and benchmarking convenience, the CLI accepts the plaintext target password, validates its length and alphabet, and internally computes its 128-bit MD5 digest.
  - The plaintext is **not** exposed to the search algorithm; candidate prefixes are tested strictly against the target digest once they reach target length $L$ (at the leaf nodes).

---

## 2. Repository Structure

```
.
├── README.md                    # Project documentation
├── Makefile                     # Build configuration for all targets
├── src/                         # Core C/CUDA source code and headers
│   ├── astar_heap.h             # Priority queue & quickselect beam trimming definitions
│   ├── astar_heap.c             # Priority queue implementation
│   ├── md5.h                    # RFC 1321 MD5 header
│   ├── md5.c                    # CPU RFC 1321 MD5 implementation
│   ├── crack_astar.c            # Sequential baseline
│   ├── crack_astar_pthreads.c   # Pthreads implementation (atomic dynamic queue)
│   ├── crack_astar_omp.c        # OpenMP implementation (dynamic loop schedule)
│   ├── crack_astar_opencilk.c   # OpenCilk implementation (cilk_for work-stealing)
│   ├── crack_astar_cuda.cu      # CUDA implementation (batch leaf hashing on GPU)
│   └── crack_astar_hybrid.cu    # Hybrid CPU-GPU implementation (streams + pinned host RAM)
├── scripts/                     # Automated benchmarking and testing suites
│   ├── run_all_astar.sh         # CPU & pure-CUDA benchmarks across thread counts
│   ├── run_hybrid_benchmark.sh  # Hybrid thread and batch-size sweep benchmarks
│   └── run_verification_tests.sh# Automated functional verification test suite
├── docs/                        # Project report
│   ├── report.tex               # LaTeX source of the report
│   └── report.pdf               # Compiled final report
├── reports_astar/               # Benchmark execution logs (single-technique)
└── reports_hybrid/              # Benchmark execution logs (hybrid sweeps)
```

---

## 3. Parallelization Strategies

### A. Pthreads (`crack_astar_pthreads.c`)
- **Task Decomposition**: The root search space is partitioned into 2-character prefix tasks ($76^2 = 5,776$ independent tasks for $L \ge 2$).
- **Dynamic Load Balancing**: Worker threads dynamically claim tasks via an atomic counter (`atomic_fetch_add`), mitigating workload imbalance caused by varying branch exploration depths.
- **Zero Heap Contention**: Each thread manages its own private min-heap and beam-trim buffer, completely eliminating lock contention during graph expansion.
- **Early Exit**: An atomic flag (`atomic_int g_found`) terminates all workers immediately once any thread recovers the target.

### B. OpenMP (`crack_astar_omp.c`)
- Decomposes the search into the same 5,776 independent prefix tasks.
- Uses dynamic scheduling: `#pragma omp parallel for schedule(dynamic, 1)`.
- Preallocates reusable per-thread heaps indexed by `omp_get_thread_num()` to avoid repeated allocation and deallocation overhead.

### C. OpenCilk (`crack_astar_opencilk.c`)
- Uses `cilk_for` to expose fine-grained prefix sub-tasks to the OpenCilk work-stealing runtime.
- Workers dynamically steal iterations when idle, ensuring high CPU core utilization.
- Thread concurrency is configured via `--threads N` or the `CILK_NWORKERS` environment variable.

### D. CUDA (`crack_astar_cuda.cu`)
- **Heterogeneous Division of Labor**: Pointer-based graph traversal, priority queue updates, and heap operations are executed on the CPU, while compute-dense MD5 leaf verification is offloaded to the GPU.
- **Batch Verification**: Leaf candidate strings are buffered into a contiguous host array.
- **Kernel Dispatch**: Once the buffer fills (or `MAX_NODES_BETWEEN_FLUSHES = 200,000` is reached), candidates are transferred to the GPU via DMA and verified concurrently using an optimized MD5 CUDA device kernel.
- `--threads BATCH_SIZE` controls the GPU buffer size (default: 1,048,576 candidates).

### E. Hybrid Pthreads + CUDA (`crack_astar_hybrid.cu`)
- Combines multi-threaded CPU dynamic task distribution with asynchronous GPU batch processing.
- Each POSIX worker thread is assigned:
  1. An independent **CUDA stream** (`cudaStreamCreate`) enabling concurrent multi-kernel execution and overlapping host-to-device memory transfers.
  2. Dedicated **pinned host memory** (`cudaHostAlloc`) for high-throughput asynchronous DMA transfers (`cudaMemcpyAsync`).
  3. Private device input/output buffers and termination flags to avoid any device-side mutexes or synchronization stalls.
- Configurable per-thread worker count (`--threads`) and per-thread GPU batch buffer size (`--batch-size`).

---

## 4. Build Instructions

### Prerequisites
* **C Compiler**: GCC or Clang with C11 and POSIX threads (`-pthread`, `-lm`).
* **OpenMP**: GCC with `-fopenmp` support.
* **OpenCilk**: Clang with OpenCilk extension enabled (`-fopencilk`).
* **NVIDIA CUDA Toolkit**: `nvcc` and an NVIDIA GPU with appropriate compute capability.

### Compilation Commands

```bash
# 1. Build standard CPU targets (Sequential, Pthreads, OpenMP)
make all

# 2. Build OpenCilk target
# (Defaults to /opt/opencilk/bin/clang; override CILK_CC if installed elsewhere or on PATH)
make crack_astar_opencilk
# or:
make crack_astar_opencilk CILK_CC=clang

# 3. Build CUDA & Hybrid targets
# Adjust NVCC_ARCH to match your GPU architecture (default is sm_70):
#   sm_70: Volta (V100)
#   sm_75: Turing (RTX 20-series, GTX 16-series)
#   sm_80 / sm_86: Ampere (A100, RTX 30-series)
#   sm_89: Ada Lovelace (RTX 40-series)
make crack_astar_cuda NVCC_ARCH=sm_89
make crack_astar_hybrid NVCC_ARCH=sm_89

# Clean all generated binaries
make clean
```

---

## 5. Usage & Examples

### Command Syntax

```bash
# Sequential baseline
./crack_astar <length> <target_password>

# Pthreads
./crack_astar_pthreads <length> <target_password> [--threads N]

# OpenMP
./crack_astar_omp <length> <target_password> [--threads N]

# OpenCilk
./crack_astar_opencilk <length> <target_password> [--threads N]

# Pure CUDA (Note: --threads specifies GPU batch size)
./crack_astar_cuda <length> <target_password> [--threads BATCH_SIZE]

# Hybrid Pthreads + CUDA
./crack_astar_hybrid <length> <target_password> [--threads N] [--batch-size BATCH_SIZE]
```

### Parameter Defaults
| Parameter | Default Value | Description |
|---|---|---|
| `--threads` (CPU targets & Hybrid) | Hardware concurrency (`sysconf(_SC_NPROCESSORS_ONLN)`) | Number of worker threads. |
| `--threads` (Pure CUDA) | `1,048,576` candidates | GPU verification batch capacity. |
| `--batch-size` (Hybrid) | `65,536` candidates | Per-thread GPU verification batch capacity. |

### Shell Escaping Notice
> [!IMPORTANT]
> The supported 76-character alphabet contains shell metacharacters such as `&`, `)`, `$`, `#`, `*`, `!`.
> **Always wrap target passwords in single quotes (`'...'`)** to prevent Bash from interpreting them as background commands, comments, or variable expansions:
> ```bash
> # Recommended:
> ./crack_astar 4 'j&mU'
> ./crack_astar_pthreads 4 '=h#B' --threads 8
> ./crack_astar_hybrid 5 'yJuq)' --threads 16 --batch-size 262144
> ```

### Practical Execution Examples

```bash
# Crack length-4 password using 8 Pthreads
./crack_astar_pthreads 4 'PtYg' --threads 8

# Crack length-4 password using OpenMP with 16 threads
./crack_astar_omp 4 'el31' --threads 16

# Crack length-4 password using OpenCilk
./crack_astar_opencilk 4 'iEl(' --threads 8

# Crack length-4 password using pure CUDA (262,144 batch size)
./crack_astar_cuda 4 'PtYg' --threads 262144

# Crack challenging length-5 password using Hybrid (16 threads, 262,144 batch size)
./crack_astar_hybrid 5 'yJuq)' --threads 16 --batch-size 262144
```

---

## 6. Automated Benchmarking Suites

The `scripts/` directory contains bash scripts to reproduce all experiment sweeps:

1. **CPU & Pure CUDA Benchmarks**:
   ```bash
   ./scripts/run_all_astar.sh [output_directory]
   ```
   Iterates through thread configurations ($1, 2, 4, 8, 16, 28, 32$) across single-technique targets and saves structured logs to `reports_astar/`.

2. **Hybrid Scaling Benchmark**:
   ```bash
   ./scripts/run_hybrid_benchmark.sh [output_directory]
   ```
   Sweeps thread counts ($1, 2, 4, 8, 16, 32$) and GPU batch sizes ($16\text{k}, 64\text{k}, 256\text{k}, 1\text{M}$) across length-4 and length-5 targets, logging metrics to `reports_hybrid/`.

3. **Automated Verification Suite**:
   ```bash
   ./scripts/run_verification_tests.sh
   ```
   Verifies functional correctness of all built implementations against reference targets.
