/*
 * GPU Cohort Agent — NVSHMEM-based Cohort FIFO participant
 *
 * Uses the same FIFO layout and head/tail semantics as cohort_fifo.h.
 * Runs persistently on the GPU, popping from input FIFOs and pushing to
 * output FIFOs via NVSHMEM symmetric memory and atomics. No FIFO spec,
 * RTL, driver, MMIO, or interrupt changes; communication is memory-centric.
 *
 * Build: nvcc -o cohort_agent_nvshmem cohort_agent_nvshmem.cu -lnvshmem -lcuda
 * Run:  mpirun -np 2 ./cohort_agent_nvshmem   (PE 0 = GPU agent, PE 1 = partner)
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

/* Same FIFO layout and semantics as cohort_fifo.h (via shared layout header) */
#include "cohort_fifo_layout.h"

#ifndef FIFO_LEN
#define FIFO_LEN                  65
#endif
#ifndef BACKOFF_ITERS
#define BACKOFF_ITERS             256
#endif

/* Total symmetric block size for one FIFO: head(8) + tail(8) + meta(16) + data */
#define COHORT_FIFO_SYM_SIZE      (COHORT_FIFO_DATA_OFFSET + (size_t)(FIFO_LEN) * COHORT_EL_SIZE_BYTES)

/* Round up to cache line for alignment */
#define COHORT_FIFO_SYM_SIZE_ALIGNED  (((COHORT_FIFO_SYM_SIZE + COHORT_CACHELINE_WIDTH - 1) / COHORT_CACHELINE_WIDTH) * COHORT_CACHELINE_WIDTH)

/* Producer PE for input FIFO (GPU pops from there); consumer PE for output FIFO (GPU pushes there) */
#ifndef INPUT_FIFO_PE
#define INPUT_FIFO_PE             1
#endif
#ifndef OUTPUT_FIFO_PE
#define OUTPUT_FIFO_PE            0
#endif

/* ========== Device: pop one element from remote input FIFO ========== */
__device__ __forceinline__ int cohort_fifo_pop_device(
    void *input_fifo_sym,
    int producer_pe,
    uint64_t *out_element)
{
    char *base = (char *)input_fifo_sym;
    uint64_t *head_ptr = (uint64_t *)(base + COHORT_FIFO_HEAD_OFFSET);
    uint64_t *tail_ptr = (uint64_t *)(base + COHORT_FIFO_TAIL_OFFSET);
    char *data_base = base + COHORT_FIFO_DATA_OFFSET;

    uint64_t tail = nvshmem_uint64_atomic_fetch(tail_ptr, producer_pe);
    uint64_t head = nvshmem_uint64_atomic_fetch(head_ptr, producer_pe);
    if (tail == head)
        return 0; /* empty */

    uint64_t pos = head & 0xFFFFFFFFULL;
    /* Read element from producer's data array; ensure ordering per cohort_fifo.h */
    nvshmem_getmem(out_element, data_base + pos * COHORT_EL_SIZE_BYTES, COHORT_EL_SIZE_BYTES, producer_pe);
    nvshmem_fence();
    /* Advance head so producer sees consumption (memory ordering as in fifo_pop_sync) */
    nvshmem_uint64_atomic_set(head_ptr, pos + 1, producer_pe);
    return 1;
}

/* ========== Device: get FIFO length (same on all PEs after init) ========== */
__device__ __forceinline__ uint32_t cohort_fifo_len_device(void *fifo_sym)
{
    uint32_t *meta_len = (uint32_t *)((char *)fifo_sym + COHORT_FIFO_META_LEN);
    return *meta_len;
}

/* ========== Device: push one element to output FIFO (local or remote) ========== */
__device__ __forceinline__ int cohort_fifo_push_device(
    void *output_fifo_sym,
    int consumer_pe,
    uint64_t element,
    uint32_t fifo_len)
{
    char *base = (char *)output_fifo_sym;
    uint64_t *head_ptr = (uint64_t *)(base + COHORT_FIFO_HEAD_OFFSET);
    uint64_t *tail_ptr = (uint64_t *)(base + COHORT_FIFO_TAIL_OFFSET);
    char *data_base = base + COHORT_FIFO_DATA_OFFSET;

    uint32_t len = fifo_len;
    uint64_t head = nvshmem_uint64_atomic_fetch(head_ptr, consumer_pe);
    uint64_t tail = nvshmem_uint64_atomic_fetch(tail_ptr, consumer_pe);
    /* Full: (tail + 1) % len == head. For non-wrap: tail+1 != head && (tail+1) % len != head */
    uint64_t next_tail = (tail + 1) & 0xFFFFFFFFULL;
    if (next_tail >= (uint64_t)len)
        next_tail = 0;
    if (next_tail == (head & 0xFFFFFFFFULL))
        return 0; /* full */

    uint64_t pos = tail & 0xFFFFFFFFULL;
    nvshmem_putmem(data_base + pos * COHORT_EL_SIZE_BYTES, &element, COHORT_EL_SIZE_BYTES, consumer_pe);
    nvshmem_fence();
    /* Advance tail so consumer sees new element (ordering as in fifo_push_sync) */
    nvshmem_uint64_atomic_set(tail_ptr, next_tail, consumer_pe);
    return 1;
}

/* ========== Device: trivial “compute” (e.g. pass-through or checksum) ========== */
__device__ __forceinline__ uint64_t cohort_agent_compute(uint64_t input)
{
    (void)input;
    return input; /* pass-through; replace with real work */
}

/* ========== Persistent GPU agent kernel ========== */
__global__ void cohort_agent_kernel(
    void *input_fifo_sym,
    void *output_fifo_sym,
    int input_pe,
    int output_pe,
    volatile int *run)
{
    int me = nvshmem_my_pe();
    if (me != OUTPUT_FIFO_PE)
        return;

    uint64_t elem;
    int backoff = 0;
    const int max_backoff = BACKOFF_ITERS;

    uint32_t out_len = cohort_fifo_len_device(output_fifo_sym);

    while (*run) {
        int popped = cohort_fifo_pop_device(input_fifo_sym, input_pe, &elem);
        if (popped) {
            backoff = 0;
            uint64_t result = cohort_agent_compute(elem);
            int pushed = 0;
            do {
                pushed = cohort_fifo_push_device(output_fifo_sym, output_pe, result, out_len);
                if (!pushed) {
                    for (int i = 0; i < max_backoff; i++)
                        ;
                }
            } while (!pushed && *run);
        } else {
            for (int i = 0; i < backoff; i++)
                ;
            if (backoff < max_backoff)
                backoff++;
        }
    }
}

/* ========== Host: initialize one symmetric FIFO (head=tail=0, meta, zero data) ========== */
static void host_init_symmetric_fifo(void *fifo_sym, uint32_t fifo_len)
{
    char *base = (char *)fifo_sym;
    *(uint64_t *)(base + COHORT_FIFO_HEAD_OFFSET) = 0;
    *(uint64_t *)(base + COHORT_FIFO_TAIL_OFFSET) = 0;
    *(uint64_t *)(base + COHORT_FIFO_META_ADDR)  = (uint64_t)(base + COHORT_FIFO_DATA_OFFSET);
    *(uint32_t *)(base + COHORT_FIFO_META_SIZE)  = (uint32_t)(COHORT_EL_SIZE_BYTES);
    *(uint32_t *)(base + COHORT_FIFO_META_LEN)   = fifo_len;
    memset(base + COHORT_FIFO_DATA_OFFSET, 0, (size_t)fifo_len * COHORT_EL_SIZE_BYTES);
}

int main(int argc, char **argv)
{
    int my_pe, n_pes;
    void *input_fifo_sym = nullptr;
    void *output_fifo_sym = nullptr;
    size_t fifo_sym_size = COHORT_FIFO_SYM_SIZE_ALIGNED;
    cudaStream_t stream = 0;
    int *h_run = nullptr;

    (void)argc;
    (void)argv;

    nvshmem_init();
    my_pe = nvshmem_my_pe();
    n_pes = nvshmem_n_pes();

    if (n_pes < 2) {
        if (my_pe == 0)
            fprintf(stderr, "cohort_agent_nvshmem: need at least 2 PEs (got %d). Use mpirun -np 2 ...\n", n_pes);
        nvshmem_finalize();
        return 1;
    }

    /* Collective symmetric allocation: same layout on all PEs */
    input_fifo_sym  = nvshmem_malloc(fifo_sym_size);
    output_fifo_sym = nvshmem_malloc(fifo_sym_size);
    if (!input_fifo_sym || !output_fifo_sym) {
        if (my_pe == 0)
            fprintf(stderr, "cohort_agent_nvshmem: nvshmem_malloc failed\n");
        nvshmem_finalize();
        return 1;
    }

    host_init_symmetric_fifo(input_fifo_sym, (uint32_t)FIFO_LEN);
    host_init_symmetric_fifo(output_fifo_sym, (uint32_t)FIFO_LEN);

    /* Run flag: device-visible so persistent kernel can poll and host can signal stop */
    h_run = (int *)nvshmem_malloc(sizeof(int));
    if (!h_run) {
        if (my_pe == 0)
            fprintf(stderr, "cohort_agent_nvshmem: nvshmem_malloc run flag failed\n");
        nvshmem_free(input_fifo_sym);
        nvshmem_free(output_fifo_sym);
        nvshmem_finalize();
        return 1;
    }
    *h_run = 1;

    if (my_pe == OUTPUT_FIFO_PE) {
        cudaSetDevice(0);
        nvshmemx_cumodule_init(nullptr); /* init device state for current CUDA context */
        /* Device reads run via symmetric ptr (same PE = local GPU address) */
        volatile int *d_run = (volatile int *)nvshmem_ptr(h_run, my_pe);
        if (d_run == nullptr) {
            fprintf(stderr, "cohort_agent_nvshmem: nvshmem_ptr(run) failed on PE %d\n", my_pe);
            nvshmem_free(h_run);
            nvshmem_free(output_fifo_sym);
            nvshmem_free(input_fifo_sym);
            nvshmem_finalize();
            return 1;
        }
        cohort_agent_kernel<<<1, 1, 0, stream>>>(
            input_fifo_sym,
            output_fifo_sym,
            INPUT_FIFO_PE,
            OUTPUT_FIFO_PE,
            d_run);
    }

    nvshmem_barrier_all();

    /* Example: run for a short time then signal device to stop */
    if (my_pe == OUTPUT_FIFO_PE) {
        for (volatile int i = 0; i < 50000000; i++)
            ;
        *h_run = 0;
        /* Ensure device sees the update (symmetric heap may be in device memory) */
        void *d_run_ptr = nvshmem_ptr(h_run, my_pe);
        if (d_run_ptr) {
            int zero = 0;
            cudaMemcpy(d_run_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice);
        }
    }
    nvshmem_barrier_all();

    if (my_pe == OUTPUT_FIFO_PE)
        cudaStreamSynchronize(stream);

    nvshmem_free(h_run);
    nvshmem_free(output_fifo_sym);
    nvshmem_free(input_fifo_sym);
    nvshmem_finalize();
    return 0;
}
