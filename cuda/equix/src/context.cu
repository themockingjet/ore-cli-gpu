/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdlib.h>
#include <equix.h>
#include "context.h"
#include "solver_heap.h"

// TODO hash_func context can be removed from here

equix_ctx* equix_alloc(equix_ctx_flags flags) {
    equix_ctx* ctx = NULL;
    
    // Allocate unified memory for equix_ctx
    if (cudaHostAlloc(&ctx, sizeof(equix_ctx), cudaHostAllocDefault) != cudaSuccess) {
        return NULL; // Directly return NULL on failure
    }
    
    // Initialize ctx to avoid potential issues
    memset(ctx, 0, sizeof(equix_ctx));
    
    ctx->flags = flags; // Set flags early to reflect the actual state
    
    ctx->hash_func = hashx_alloc(flags & EQUIX_CTX_COMPILE ?
        HASHX_COMPILED : HASHX_INTERPRETED);
    if (ctx->hash_func == NULL || ctx->hash_func == HASHX_NOTSUPP) {
        equix_free(ctx); // Free resources before returning
        return NULL; // Use NULL directly for clarity
    }
    
    if (flags & EQUIX_CTX_SOLVE) {
        if (cudaHostAlloc(&ctx->heap, sizeof(solver_heap), cudaHostAllocDefault) != cudaSuccess) {
            equix_free(ctx); // Free resources before returning
            return NULL;
        }
    }
    
    return ctx;
}

void equix_free(equix_ctx* ctx) {
	if (ctx != NULL && ctx != EQUIX_NOTSUPP) {
		if (ctx->flags & EQUIX_CTX_SOLVE) {
			cudaFreeHost(ctx->heap);
		}
		hashx_free(ctx->hash_func);
		cudaFreeHost(ctx);
	}
}