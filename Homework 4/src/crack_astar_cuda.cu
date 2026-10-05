/*
 * crack_astar_cuda.cu
 * CUDA implementation of A* password recovery with GPU batch leaf verification.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include "astar_heap.h"
#include "md5.h"

#define MAX_CHARSET_LEN 128
#define THREADS_PER_BLOCK 256
#define DEFAULT_BATCH_SIZE 1048576ULL

/* Flush threshold for batched GPU leaf testing */
#define MAX_NODES_BETWEEN_FLUSHES 200000ULL

#define HEURISTIC_WEIGHT 1.0  /* Admissible heuristic weight for uniform costs */
#define TIME_BUDGET_SECONDS 60.0
#define TIME_CHECK_INTERVAL 5000
#define NODE_BUDGET 30000000ULL

typedef unsigned long long u64;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

static char g_charset[MAX_CHARSET_LEN];
static int g_charset_len;
static int g_char_to_idx[256];

/* Uniform cost per character transition */
static const double EDGE_COST = 1.0;

/* GPU batch-testing kernel */
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

__device__ int d_found_flag;
__device__ int d_found_batch_index;
__constant__ unsigned char d_target_digest[16];

__device__ inline void d_md5_single_block(const unsigned char *msg, int len, unsigned char *digest) {
    static const unsigned int r[64] = {
        7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
        5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
        4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
        6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
    };
    static const unsigned int k[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,
        0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
        0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,
        0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,
        0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
        0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,
        0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,
        0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
        0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
    };

    unsigned char block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < len; i++) block[i] = msg[i];
    block[len] = 0x80;
    unsigned long long bits_len = (unsigned long long)len * 8ULL;
    for (int i = 0; i < 8; i++) block[56 + i] = (unsigned char)((bits_len >> (8 * i)) & 0xFF);

    unsigned int w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = (unsigned int)block[i*4] | ((unsigned int)block[i*4+1] << 8)
             | ((unsigned int)block[i*4+2] << 16) | ((unsigned int)block[i*4+3] << 24);
    }

    unsigned int h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
    unsigned int a = h0, b = h1, c = h2, d = h3;

    for (unsigned int i = 0; i < 64; i++) {
        unsigned int f, g;
        if (i < 16)      { f = (b & c) | (~b & d);  g = i; }
        else if (i < 32) { f = (d & b) | (~d & c);  g = (5*i + 1) % 16; }
        else if (i < 48) { f = b ^ c ^ d;            g = (3*i + 5) % 16; }
        else             { f = c ^ (b | ~d);          g = (7*i) % 16; }
        unsigned int temp = d;
        d = c; c = b;
        b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
        a = temp;
    }
    h0 += a; h1 += b; h2 += c; h3 += d;
    memcpy(digest, &h0, 4); memcpy(digest + 4, &h1, 4);
    memcpy(digest + 8, &h2, 4); memcpy(digest + 12, &h3, 4);
}

__device__ inline int d_md5_equal(const unsigned char *a, const unsigned char *b) {
    for (int i = 0; i < 16; i++) if (a[i] != b[i]) return 0;
    return 1;
}

/* d_batch holds batch_count candidates, each a fixed (MAX_LEN+1)-byte
 * stride (nul-terminated string, unused trailing bytes don't matter).
 * One thread tests exactly one candidate -- this is the embarrassingly
 * parallel part of A*: verifying a set of already-decided-on complete
 * candidates has no sequential dependency between them at all. */
__global__ void batch_test_kernel(const char *d_batch, int batch_count, int cand_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    const char *candidate = d_batch + (size_t)idx * (MAX_LEN + 1);
    unsigned char digest[16];
    d_md5_single_block((const unsigned char *)candidate, cand_len, digest);

    if (d_md5_equal(digest, d_target_digest)) {
        int expected = 0;
        if (atomicCAS(&d_found_flag, expected, 1) == expected) {
            d_found_batch_index = idx;
        }
    }
}

/* Host-side wrapper: copies the batch to the GPU, launches the kernel,
 * checks the result. Returns 1 and fills found_password if a match was
 * found in this batch, 0 otherwise. d_batch_buf is a persistent device
 * buffer allocated once in main() and reused across calls, to avoid
 * repeated cudaMalloc/cudaFree overhead on every batch. */
static int flush_batch_to_gpu(char *host_batch, int batch_count, int cand_len,
                               char *d_batch_buf, char *found_password) {
    if (batch_count == 0) return 0;

    CUDA_CHECK(cudaMemcpy(d_batch_buf, host_batch, (size_t)batch_count * (MAX_LEN + 1),
                          cudaMemcpyHostToDevice));

    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_found_flag, &zero, sizeof(int)));

    int blocks = (batch_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    batch_test_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_batch_buf, batch_count, cand_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int found_flag_host;
    CUDA_CHECK(cudaMemcpyFromSymbol(&found_flag_host, d_found_flag, sizeof(int)));
    if (found_flag_host) {
        int found_idx_host;
        CUDA_CHECK(cudaMemcpyFromSymbol(&found_idx_host, d_found_batch_index, sizeof(int)));
        memcpy(found_password, host_batch + (size_t)found_idx_host * (MAX_LEN + 1), (size_t)cand_len);
        found_password[cand_len] = '\0';
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <length> <target_password> [--threads N]\n"
                         "  --threads N : batch size -- how many complete-length candidates\n"
                         "                accumulate before being tested on the GPU in one\n"
                         "                parallel launch (default %llu). See file header for\n"
                         "                why very small batch sizes may not help.\n",
                argv[0], (unsigned long long)DEFAULT_BATCH_SIZE);
        return 1;
    }

    int length = atoi(argv[1]);
    const char *target_password = argv[2];
    u64 batch_size = DEFAULT_BATCH_SIZE;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            batch_size = strtoull(argv[++i], NULL, 10);
        }
    }
    if (batch_size < 1) batch_size = 1;

    if (length <= 0 || length > MAX_LEN) {
        fprintf(stderr, "Error: length must be between 1 and %d\n", MAX_LEN);
        return 1;
    }
    if ((int)strlen(target_password) != length) {
        fprintf(stderr, "Error: target password length must equal <length>\n");
        return 1;
    }

    strcpy(g_charset, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+");
    g_charset_len = (int)strlen(g_charset);
    for (int i = 0; i < 256; i++) g_char_to_idx[i] = -1;
    for (int i = 0; i < g_charset_len; i++) g_char_to_idx[(unsigned char)g_charset[i]] = i;

    for (int i = 0; i < length; i++) {
        if (g_char_to_idx[(unsigned char)target_password[i]] < 0) {
            fprintf(stderr, "Error: target password contains a character outside the supported alphabet\n");
            return 1;
        }
    }

    unsigned char target_digest[16];
    md5((const unsigned char *)target_password, (size_t)length, target_digest);

    char target_hex[33];
    md5_to_hex(target_digest, target_hex);

    printf("Password len  : %d\n", length);
    printf("Charset       : %d symbols (fixed, not selectable)\n", g_charset_len);
    printf("Cost model    : uniform\n");
    printf("Target MD5    : %s\n", target_hex);
    printf("Batch size    : %llu (leaf candidates tested per GPU launch)\n", (unsigned long long)batch_size);
    printf("============================================================\n");
    fflush(stdout);

    /* Persistent device buffer + constant target digest, set up once. */
    char *d_batch_buf;
    CUDA_CHECK(cudaMalloc(&d_batch_buf, batch_size * (MAX_LEN + 1)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_target_digest, target_digest, 16));

    char *host_batch = (char *)malloc(batch_size * (MAX_LEN + 1));
    u64 batch_count = 0;
    u64 gpu_batches_launched = 0;
    double gpu_time_total = 0.0;

    Heap heap;
    heap_init(&heap, 1024);
    State start;
    start.depth = 0; start.g = 0.0;
    start.f = length * HEURISTIC_WEIGHT * EDGE_COST;
    start.prefix[0] = '\0';
    heap_push(&heap, start);

    u64 nodes_expanded = 0, leaf_tests = 0;
    u64 nodes_since_last_flush = 0;
    int found = 0;
    char found_password[MAX_LEN + 1] = {0};

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (heap.size > 0) {
        State s = heap_pop(&heap);

        if (s.depth == length) {
            leaf_tests++;
            memcpy(host_batch + batch_count * (MAX_LEN + 1), s.prefix, (size_t)length + 1);
            batch_count++;

            if (batch_count == batch_size) {
                struct timespec g0, g1;
                clock_gettime(CLOCK_MONOTONIC, &g0);
                int hit = flush_batch_to_gpu(host_batch, (int)batch_count, length, d_batch_buf, found_password);
                clock_gettime(CLOCK_MONOTONIC, &g1);
                gpu_time_total += (g1.tv_sec - g0.tv_sec) + (g1.tv_nsec - g0.tv_nsec) / 1e9;
                gpu_batches_launched++;
                batch_count = 0;
                nodes_since_last_flush = 0;
                if (hit) { found = 1; break; }
            }
            continue;
        }

        nodes_expanded++;
        nodes_since_last_flush++;

        for (int c = 0; c < g_charset_len; c++) {
            double step_cost = EDGE_COST;
            State child;
            child.depth = s.depth + 1;
            memcpy(child.prefix, s.prefix, (size_t)s.depth);
            child.prefix[s.depth] = g_charset[c];
            child.prefix[s.depth + 1] = '\0';
            child.g = s.g + step_cost;
            child.f = child.g + HEURISTIC_WEIGHT * (length - child.depth) * EDGE_COST;
            heap_push(&heap, child);
        }

        heap_trim_if_needed(&heap);

        if (nodes_since_last_flush >= MAX_NODES_BETWEEN_FLUSHES && batch_count > 0) {
            struct timespec g0, g1;
            clock_gettime(CLOCK_MONOTONIC, &g0);
            int hit = flush_batch_to_gpu(host_batch, (int)batch_count, length, d_batch_buf, found_password);
            clock_gettime(CLOCK_MONOTONIC, &g1);
            gpu_time_total += (g1.tv_sec - g0.tv_sec) + (g1.tv_nsec - g0.tv_nsec) / 1e9;
            gpu_batches_launched++;
            batch_count = 0;
            nodes_since_last_flush = 0;
            if (hit) { found = 1; break; }
        }

        if (nodes_expanded % TIME_CHECK_INTERVAL == 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed_so_far = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
            if (elapsed_so_far >= TIME_BUDGET_SECONDS) {
                printf("    A* time budget (%.0fs) reached -- stopping.\n", TIME_BUDGET_SECONDS);
                break;
            }
        }
        if (nodes_expanded >= NODE_BUDGET) {
            printf("    A* node budget reached -- stopping.\n");
            break;
        }
    }

    /* Flush remaining partial batch */
    if (!found && batch_count > 0) {
        struct timespec g0, g1;
        clock_gettime(CLOCK_MONOTONIC, &g0);
        int hit = flush_batch_to_gpu(host_batch, (int)batch_count, length, d_batch_buf, found_password);
        clock_gettime(CLOCK_MONOTONIC, &g1);
        gpu_time_total += (g1.tv_sec - g0.tv_sec) + (g1.tv_nsec - g0.tv_nsec) / 1e9;
        gpu_batches_launched++;
        if (hit) found = 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("    Search: %llu nodes expanded, %llu leaf tests\n", nodes_expanded, leaf_tests);
    printf("    GPU: %llu batch(es) launched, %.3fs total GPU time\n", gpu_batches_launched, gpu_time_total);
    printf("============================================================\n");
    if (found) {
        printf("CRACKED       : %s\n", found_password);
    } else {
        printf("Password NOT recovered.\n");
    }
    printf("Elapsed time   : %.6f seconds\n", total_time);
    printf("============================================================\n");

    free(host_batch);
    cudaFree(d_batch_buf);
    heap_free(&heap);
    return 0;
}
