# ==========================================
# Makefile for Hybrid MPI+Cilk Project
# Connected Components on Massive Graphs
# ==========================================

# 1. Compiler Selection
# mpicc is the standard wrapper for MPI programs
CC = mpicc

# 2. Compilation Flags
# -O3: Enables high-level optimization (crucial for graph processing)
# -Wall: Enables all compiler warnings
# -fopencilk: Flag for the OpenCilk/Cilk Plus compiler
# -g: Includes debug information (optional)
CFLAGS = -O3 -Wall -fopencilk

# 3. Target and Source Files
TARGET = cc_hybrid
SRC = main.c
OBJ = $(SRC:.c=.o)

# 4. Default Rule
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 5. Clean Rule
# Removes the executable and object files to allow a fresh build
clean:
	rm -f $(TARGET) $(OBJ)

# 6. Helper Run Rule
# Usage: make run NODES=2 FILE=/path/to/graph.mtx
NODES ?= 1
FILE ?= sample.mtx
run: $(TARGET)
	mpirun -np $(NODES) ./$(TARGET) $(FILE)

.PHONY: all clean run