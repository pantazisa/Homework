# Compilers
CC = gcc
# Note: Adjust CILK_CC path if OpenCilk clang is located elsewhere (e.g., /opencilk/bin/clang)
CILK_CC = clang 

# Flags

CFLAGS = -Wall

PTHREAD_FLAGS = -lpthread
OMP_FLAGS = -fopenmp
CILK_FLAGS = -fopencilk

# Targets
TARGETS = ccomponents ccpthreads ccopenmp ccopencilk

# Default target: Build all
all: $(TARGETS)

# Sequential Version
ccomponents: ccomponents.c
	$(CC) $(CFLAGS) -o ccomponents ccomponents.c

# Pthreads Version
ccpthreads: ccpthreads.c
	$(CC) $(CFLAGS) ccpthreads.c -o ccpthreads $(PTHREAD_FLAGS)

# OpenMP Version
ccopenmp: ccopenmp.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o ccopenmp ccopenmp.c

# OpenCilk Version
ccopencilk: ccopencilk.c
	$(CILK_CC) $(CFLAGS) $(CILK_FLAGS) -o ccopencilk ccopencilk.c

# Clean
clean:
	rm -f $(TARGETS) *.bin

.PHONY: all clean