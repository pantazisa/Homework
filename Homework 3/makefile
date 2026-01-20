NVCC = nvcc
CFLAGS = -O3 -Xcompiler -Wall
ARCH = -arch=sm_89
TARGET = cc_cuda_final
SRC = cc_cuda_final.cu

all: $(TARGET1) $(TARGET2)
 
$(TARGET1): cc_cuda_v1.cu
	$(NVCC) $(CFLAGS) $(ARCH) cc_cuda_v1.cu -o $(TARGET1)

$(TARGET2): cc_cuda_v2.cu
	$(NVCC) $(CFLAGS) $(ARCH) cc_cuda_v2.cu -o $(TARGET2)

clean:
	rm -f $(TARGET1) $(TARGET2) benchmarks.csv *.bin

run_v1: $(TARGET1)
	./$(TARGET1) $(FILE)

run_v2: $(TARGET2)
	./$(TARGET2) $(FILE)

benchmark: $(TARGET1) $(TARGET2)
	@for i in $$(seq 1 10); do \
		./$(TARGET1) com-Friendster/com-Friendster.mtx; \
	done
	@for i in $$(seq 1 10); do \
		./$(TARGET1) mawi_201512020330.mtx; \
	done

	@for i in $$(seq 1 10); do \
		./$(TARGET2) com-Friendster/com-Friendster.mtx; \
	done
	@for i in $$(seq 1 10); do \
		./$(TARGET2) mawi_201512020330.mtx; \
	done