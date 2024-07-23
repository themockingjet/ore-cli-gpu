#include <stdint.h>
#include <stdio.h>
#include "drillx.h"
#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

const int BATCH_SIZE = 256;

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out) {
    // Allocate pinned memory for ctxs and hash_space
    hashx_ctx** ctxs;
    uint64_t** hash_space;

    cudaHostAlloc(&ctxs, BATCH_SIZE * sizeof(hashx_ctx*), cudaHostAllocDefault);
    cudaHostAlloc(&hash_space, BATCH_SIZE * sizeof(uint64_t*), cudaHostAllocDefault);

    for (int i = 0; i < BATCH_SIZE; i++) {
        cudaMalloc(&hash_space[i], INDEX_SPACE * sizeof(uint64_t));
    }

    // Prepare seed and hash contexts
    uint8_t seed[40];
    memcpy(seed, challenge, 32);
    for (int i = 0; i < BATCH_SIZE; i++) {
        uint64_t nonce_offset = *((uint64_t*)nonce) + i;
        memcpy(seed + 32, &nonce_offset, 8);
        ctxs[i] = hashx_alloc(HASHX_INTERPRETED);
        if (!ctxs[i] || !hashx_make(ctxs[i], seed, 40)) {
            cudaFreeHost(hash_space);
            cudaFreeHost(ctxs);
            return;
        }
    }

    // Launch kernel to parallelize hashx operations
    dim3 threadsPerBlock(256); // 256 threads per block
    dim3 blocksPerGrid((65536 * BATCH_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x); // enough blocks to cover batch
    do_hash_stage0i<<<blocksPerGrid, threadsPerBlock>>>(ctxs, hash_space);
    cudaDeviceSynchronize();

    // Copy hashes back to cpu
    for (int i = 0; i < BATCH_SIZE; i++) {
        cudaMemcpy(out + i * INDEX_SPACE, hash_space[i], INDEX_SPACE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // Free memory
    for (int i = 0; i < BATCH_SIZE; i++) {
        hashx_free(ctxs[i]);
        cudaFree(hash_space[i]);
    }
    cudaFreeHost(hash_space);
    cudaFreeHost(ctxs);
}

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space) {
    uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = item / INDEX_SPACE;
    uint32_t i = item % INDEX_SPACE;
    if (batch_idx < BATCH_SIZE) {
        hash_stage0i(ctxs[batch_idx], hash_space[batch_idx], i);
    }
}

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols, int num_sets) {
    // Allocate device memory
    uint64_t *d_hashes;
    solver_heap *d_heaps;
    equix_solution *d_solutions;
    uint32_t *d_num_sols;

    cudaMalloc(&d_hashes, num_sets * INDEX_SPACE * sizeof(uint64_t));
    cudaMalloc(&d_heaps, num_sets * sizeof(solver_heap));
    cudaMalloc(&d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution));
    cudaMalloc(&d_num_sols, num_sets * sizeof(uint32_t));

    // Copy input data to device
    cudaMemcpy(d_hashes, hashes, num_sets * INDEX_SPACE * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_sets + threadsPerBlock - 1) / threadsPerBlock;
    solve_all_stages_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_heaps, d_solutions, d_num_sols);

    // Copy results back to host
    equix_solution *h_solutions = new equix_solution[num_sets * EQUIX_MAX_SOLS];
    uint32_t *h_num_sols = new uint32_t[num_sets];

    cudaMemcpy(h_solutions, d_solutions, num_sets * EQUIX_MAX_SOLS * sizeof(equix_solution), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_sols, d_num_sols, num_sets * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Process results
    for (int i = 0; i < num_sets; i++) {
        sols[i] = h_num_sols[i];
        if (h_num_sols[i] > 0) {
            memcpy(out + i * sizeof(equix_solution), &h_solutions[i * EQUIX_MAX_SOLS], sizeof(equix_solution));
        }
    }

    // Free device memory
    cudaFree(d_hashes);
    cudaFree(d_heaps);
    cudaFree(d_solutions);
    cudaFree(d_num_sols);

    // Free host memory
    delete[] h_solutions;
    delete[] h_num_sols;
}