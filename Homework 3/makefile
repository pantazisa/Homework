NVCC = nvcc
CFLAGS = -O3 -Xcompiler -Wall
ARCH = -arch=sm_86
TARGET = cc_cuda_final
SRC = cc_cuda_final.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(ARCH) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) benchmarks.csv

run: $(TARGET)
	./$(TARGET) $(FILE)

benchmark: $(TARGET)
	@for i in $$(seq 1 10); do \
		./$(TARGET) com-Friendster/com-Friendster.mtx; \
	done
	@fore i in $$(seq 1 10); do \
		./$(TARGET) mawi_201512020330.mtx
	done
