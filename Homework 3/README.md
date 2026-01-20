This `README.md` provides clear instructions for building the project, running the different versions (V1 and V2), and replicating the benchmarks used in the report.

***

# High-Performance Connected Components with CUDA

This repository contains two parallel implementations of the Connected Components (CC) algorithm for large-scale undirected graphs using NVIDIA CUDA.

- **V1 (Baseline):** A naive implementation using one thread per vertex.
- **V2 (Optimized):** A high-performance Warp-Centric implementation using thread collaboration, warp shuffles, and memory-mapped I/O.

## Prerequisites

### Hardware
*   **NVIDIA GPU:** Architecture Maxwell (sm_35) or newer. Optimized for Ampere/Ada (sm_86/sm_89).
*   **VRAM:** At least 4GB for medium graphs (mawi). 16GB+ recommended for large graphs (com-Friendster).
*   **System RAM:** Sufficient RAM to load the `.mtx` file (approx. 16GB for Friendster).

### Software
*   **CUDA Toolkit:** `nvcc` compiler installed.
*   **Linux/WSL:** Required for `mmap` and `Makefile` utilities.

## Build Instructions

The project includes a `Makefile` to automate the compilation of both versions.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pantazisa/Homework.git
    cd Homework
    ```

2.  **Compile the code:**
    By default, the Makefile targets `sm_86` (RTX 30-series). Adjust the `ARCH` variable in the Makefile if using a different GPU (e.g., `sm_75` for T4, `sm_89` for RTX 40-series).
    ```bash
    make
    ```
    This generates two executables: `cc_v1` and `cc_v2`.

## Usage

The program accepts graphs in the **Matrix Market (.mtx)** format. It automatically handles weighted graphs by skipping the weight column and caches the adjacency structure into a `.bin` file for faster subsequent loads.

### Running V1 (Naive)
```bash
./cc_v1 path/to/graph.mtx
```

### Running V2 (Optimized)
```bash
./cc_v2 path/to/graph.mtx
```

### Using the Makefile shortcuts
```bash
make run_v1 FILE=mawi_201512020330.mtx
make run_v2 FILE=mawi_201512020330.mtx
```

## Benchmarking

To replicate the results found in the report (10 iterations per graph for both versions), use the automated benchmarking tool:

```bash
make benchmark
```

Results will be printed to the console and appended to a file named `benchmarks.csv` in the following format:
`Graph, Vertices, Edges, Components, Time(s)`

## Optimization Key Features

*   **Warp-Centric Logic (V2):** Assigns 32 threads to each vertex to eliminate warp divergence on high-degree "hub" nodes.
*   **Register Shuffles:** Uses `__shfl_down_sync` for ultra-fast intra-warp communication.
*   **Pointer Jumping:** Implements path compression to achieve $O(\log D)$ convergence.
*   **mmap I/O:** Uses memory mapping and a custom integer parser to bypass the slow `sscanf` standard library function.

## Author
**Apostolos Pantazis**  
AM: 10910  
Parallel and Distributed Systems - Homework 3