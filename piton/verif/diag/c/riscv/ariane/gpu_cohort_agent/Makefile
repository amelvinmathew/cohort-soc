# GPU Cohort Agent (NVSHMEM) — same FIFO layout as cohort_fifo.h
# Requires: NVSHMEM, CUDA; run with mpirun -np 2 ./cohort_agent_nvshmem

NVCC     ?= nvcc
# Include path to cohort_fifo_layout.h (same layout as cohort_fifo.h)
COHORT_INC ?= $(dir $(CURDIR))../../../assembly/include/riscv/ariane
CFLAGS   ?= -O2 -std=c++14 -I$(COHORT_INC)
LDFLAGS  ?= -lnvshmem -lcuda

TARGET   = cohort_agent_nvshmem
SRC      = cohort_agent_nvshmem.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
