#include <stdlib.h>
#include <string.h>
#include <hashx.h>
#include "context.h"
#include "compiler.h"
#include "program.h"

#define STRINGIZE_INNER(x) #x
#define STRINGIZE(x) STRINGIZE_INNER(x)

#ifndef HASHX_SALT
#define HASHX_SALT HashX v1
#endif

__device__ const blake2b_param hashx_blake2_params = {
    64, 0, 1, 1, 0, 0, 0, 0, { 0 }, STRINGIZE(HASHX_SALT), { 0 }
};

hashx_ctx* hashx_alloc(hashx_type type) {
    hashx_ctx* ctx;
    cudaMallocManaged(&ctx, sizeof(hashx_ctx));
    
    ctx->code = NULL;
    if (type & HASHX_COMPILED) {
        if (!hashx_compiler_init(ctx)) {
            cudaFree(ctx);
            return NULL;
        }
        ctx->type = HASHX_COMPILED;
    }
    else {
        cudaMallocManaged(&ctx->program, sizeof(hashx_program));
        ctx->type = HASHX_INTERPRETED;
    }
    
#ifdef HASHX_BLOCK_MODE
    memcpy(&ctx->params, &hashx_blake2_params, 32);
#endif

    return ctx;
}

void hashx_free(hashx_ctx* ctx) {
    if (ctx != NULL && ctx != HASHX_NOTSUPP) {
        if (ctx->code != NULL) {
            if (ctx->type & HASHX_COMPILED) {
                hashx_compiler_destroy(ctx);
            }
            else {
                cudaFree(ctx->program);
            }
        }
        cudaFree(ctx);
    }
}