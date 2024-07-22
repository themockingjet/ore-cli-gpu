/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "blake2.h"
#include "hashx_endian.h"

// SIMD headers, adjust as needed for your target architecture
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static const uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

// Sigma values stored in a lookup table
static const uint8_t blake2b_sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

static FORCE_INLINE uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

static FORCE_INLINE void blake2b_set_lastblock(blake2b_state* S) {
    S->f[0] = UINT64_C(0xFFFFFFFFFFFFFFFF);
}

static FORCE_INLINE void blake2b_increment_counter(blake2b_state* S, uint64_t inc) {
    S->t[0] += inc;
    S->t[1] += (S->t[0] < inc);
}

static FORCE_INLINE void blake2b_init0(blake2b_state* S) {
    memset(S, 0, sizeof(*S));
    memcpy(S->h, blake2b_IV, sizeof(S->h));
}

int hashx_blake2b_init_param(blake2b_state* S, const blake2b_param* P) {
    if (NULL == P || NULL == S) {
        return -1;
    }

    blake2b_init0(S);
    const uint64_t* p = (const uint64_t*)P;
    for (int i = 0; i < 8; ++i) {
        S->h[i] ^= p[i];
    }
    S->outlen = P->digest_length;
    return 0;
}

#define G(r, i, a, b, c, d)                                                  \
    do {                                                                     \
        a += b + m[blake2b_sigma[r][2*i]];                                   \
        d = rotr64(d ^ a, 32);                                               \
        c += d;                                                              \
        b = rotr64(b ^ c, 24);                                               \
        a += b + m[blake2b_sigma[r][2*i+1]];                                 \
        d = rotr64(d ^ a, 16);                                               \
        c += d;                                                              \
        b = rotr64(b ^ c, 63);                                               \
    } while (0)

#define ROUND(r)                                                             \
    do {                                                                     \
        G(r, 0, v[0], v[4], v[8], v[12]);                                    \
        G(r, 1, v[1], v[5], v[9], v[13]);                                    \
        G(r, 2, v[2], v[6], v[10], v[14]);                                   \
        G(r, 3, v[3], v[7], v[11], v[15]);                                   \
        G(r, 4, v[0], v[5], v[10], v[15]);                                   \
        G(r, 5, v[1], v[6], v[11], v[12]);                                   \
        G(r, 6, v[2], v[7], v[8], v[13]);                                    \
        G(r, 7, v[3], v[4], v[9], v[14]);                                    \
    } while (0)

static void blake2b_compress(blake2b_state* S, const uint8_t* block) {
    uint64_t m[16];
    uint64_t v[16];

    for (int i = 0; i < 16; ++i) {
        m[i] = load64(block + i * sizeof(m[i]));
    }

    for (int i = 0; i < 8; ++i) {
        v[i] = S->h[i];
        v[i + 8] = blake2b_IV[i];
    }

    v[12] ^= S->t[0];
    v[13] ^= S->t[1];
    v[14] ^= S->f[0];
    v[15] ^= S->f[1];

    ROUND(0);
    ROUND(1);
    ROUND(2);
    ROUND(3);
    ROUND(4);
    ROUND(5);
    ROUND(6);
    ROUND(7);
    ROUND(8);
    ROUND(9);
    ROUND(10);
    ROUND(11);

    for (int i = 0; i < 8; ++i) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

int hashx_blake2b_update(blake2b_state* S, const void* in, size_t inlen) {
    if (inlen == 0) {
        return 0;
    }

    if (S == NULL || in == NULL || S->f[0] != 0) {
        return -1;
    }

    const uint8_t* pin = (const uint8_t*)in;

    if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
        size_t left = S->buflen;
        size_t fill = BLAKE2B_BLOCKBYTES - left;
        memcpy(&S->buf[left], pin, fill);
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress(S, S->buf);
        S->buflen = 0;
        inlen -= fill;
        pin += fill;

        while (inlen > BLAKE2B_BLOCKBYTES) {
            blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
            blake2b_compress(S, pin);
            inlen -= BLAKE2B_BLOCKBYTES;
            pin += BLAKE2B_BLOCKBYTES;
        }
    }

    memcpy(&S->buf[S->buflen], pin, inlen);
    S->buflen += (unsigned int)inlen;
    return 0;
}

int hashx_blake2b_final(blake2b_state* S, void* out, size_t outlen) {
    if (S == NULL || out == NULL || outlen < S->outlen || S->f[0] != 0) {
        return -1;
    }

    blake2b_increment_counter(S, S->buflen);
    blake2b_set_lastblock(S);
    memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen);
    blake2b_compress(S, S->buf);

    uint8_t buffer[BLAKE2B_OUTBYTES];
    for (int i = 0; i < 8; ++i) {
        store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
    }

    memcpy(out, buffer, S->outlen);
    return 0;
}

void hashx_blake2b_4r(const blake2b_param* params, const void* in, size_t inlen, void* out) {
    blake2b_state state;
    const uint64_t* p = (const uint64_t*)params;

    blake2b_init0(&state);
    for (int i = 0; i < 8; ++i) {
        state.h[i] ^= p[i];
    }

    const uint8_t* pin = (const uint8_t*)in;

    while (inlen > BLAKE2B_BLOCKBYTES) {
        blake2b_increment_counter(&state, BLAKE2B_BLOCKBYTES);
        blake2b_compress(&state, pin);
        inlen -= BLAKE2B_BLOCKBYTES;
        pin += BLAKE2B_BLOCKBYTES;
    }

    memcpy(state.buf, pin, inlen);
    blake2b_increment_counter(&state, inlen);
    blake2b_set_lastblock(&state);
    blake2b_compress(&state, state.buf);

    memcpy(out, state.h, sizeof(state.h));
}