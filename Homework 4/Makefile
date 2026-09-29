# ---------------------------------------------------------------------------
# Makefile for the A* password-cracking project
# ---------------------------------------------------------------------------
# Sources live in src/. Binaries are built in the project root.
#
# Targets you can build with gcc:
#   make crack_astar          sequential baseline
#   make crack_astar_pthreads Pthreads version (dynamic task queue)
#   make crack_astar_omp      OpenMP version (dynamic task queue)
#   make all                  builds all three of the above
#
# Targets requiring specific toolchains:
#   make crack_astar_opencilk   needs OpenCilk-enabled clang (https://www.opencilk.org/)
#   make crack_astar_cuda       needs nvcc + NVIDIA GPU
#   make crack_astar_hybrid     needs nvcc + NVIDIA GPU + pthreads
# ---------------------------------------------------------------------------

SRCDIR     ?= src

CC         ?= gcc
CFLAGS     ?= -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=199309L
CFLAGS     += -I$(SRCDIR)
LDLIBS     ?= -lm

CILK_CC    ?= /opt/opencilk/bin/clang
CILK_FLAGS ?= -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200112L -fopencilk -I$(SRCDIR)

NVCC       ?= nvcc
NVCC_ARCH  ?= sm_70
NVCC_FLAGS ?= -O3 -arch=$(NVCC_ARCH) -I$(SRCDIR)

# Shared CPU sources and headers
COMMON     = $(SRCDIR)/astar_heap.c $(SRCDIR)/md5.c
HEADERS    = $(SRCDIR)/md5.h $(SRCDIR)/astar_heap.h

all: crack_astar crack_astar_pthreads crack_astar_omp

crack_astar: $(SRCDIR)/crack_astar.c $(COMMON) $(HEADERS)
	$(CC) $(CFLAGS) -o crack_astar $(SRCDIR)/crack_astar.c $(COMMON) $(LDLIBS)

crack_astar_pthreads: $(SRCDIR)/crack_astar_pthreads.c $(COMMON) $(HEADERS)
	$(CC) $(CFLAGS) -pthread -o crack_astar_pthreads $(SRCDIR)/crack_astar_pthreads.c $(COMMON) $(LDLIBS)

crack_astar_omp: $(SRCDIR)/crack_astar_omp.c $(COMMON) $(HEADERS)
	$(CC) $(CFLAGS) -fopenmp -o crack_astar_omp $(SRCDIR)/crack_astar_omp.c $(COMMON) $(LDLIBS)

# Requires an OpenCilk release of clang on PATH (or override CILK_CC), e.g.:
#   make crack_astar_opencilk CILK_CC=/opt/opencilk/bin/clang
crack_astar_opencilk: $(SRCDIR)/crack_astar_opencilk.c $(COMMON) $(HEADERS)
	$(CILK_CC) $(CILK_FLAGS) -o crack_astar_opencilk $(SRCDIR)/crack_astar_opencilk.c $(COMMON) $(LDLIBS)

# Requires the CUDA toolkit (nvcc) and an NVIDIA GPU. Adjust NVCC_ARCH to
# match your GPU's compute capability, e.g.:
#   make crack_astar_cuda NVCC_ARCH=sm_86      (Ampere / RTX 30-series)
#   make crack_astar_cuda NVCC_ARCH=sm_89      (Ada / RTX 40-series)
crack_astar_cuda: $(SRCDIR)/crack_astar_cuda.cu $(COMMON)
	$(NVCC) $(NVCC_FLAGS) -o crack_astar_cuda $(SRCDIR)/crack_astar_cuda.cu $(COMMON)

# Requires the CUDA toolkit, an NVIDIA GPU, and pthreads (-lpthread).
crack_astar_hybrid: $(SRCDIR)/crack_astar_hybrid.cu $(COMMON)
	$(NVCC) $(NVCC_FLAGS) -o crack_astar_hybrid $(SRCDIR)/crack_astar_hybrid.cu $(COMMON) -lpthread

clean:
	rm -f crack_astar crack_astar_pthreads crack_astar_omp crack_astar_opencilk crack_astar_cuda crack_astar_hybrid

.PHONY: all clean
