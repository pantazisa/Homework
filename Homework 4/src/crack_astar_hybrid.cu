/*
 * crack_astar_hybrid.cu
 * Hybrid Pthreads + CUDA parallel implementation of A* password recovery.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <pthread.h>
#include <atomic>
#include <cuda_runtime.h>
#include "astar_heap.h"
#include "md5.h"

#define MAX_CHARSET_LEN 128
#define THREADS_PER_BLOCK 256
#define DEFAULT_BATCH_SIZE 65536ULL

#define HEURISTIC_WEIGHT 1.0  /* Admissible heuristic weight for uniform costs */
#define TIME_BUDGET_SECONDS 60.0
#define NODE_BUDGET 35000000ULL

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

static std::atomic<int> g_found_flag(0);
static char g_found_password[MAX_LEN + 1];
static pthread_mutex_t g_result_mutex = PTHREAD_MUTEX_INITIALIZER;

/* GPU batch-testing kernel */
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

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

/* found_flag/found_index are THIS THREAD's own device allocations
 * (passed in per-call), never a shared global -- see file header. */
__global__ void batch_test_kernel(const char *d_batch, int batch_count, int cand_len,
                                   int *found_flag, int *found_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    const char *candidate = d_batch + (size_t)idx * (MAX_LEN + 1);
    unsigned char digest[16];
    d_md5_single_block((const unsigned char *)candidate, cand_len, digest);

    if (d_md5_equal(digest, d_target_digest)) {
        int expected = 0;
        if (atomicCAS(found_flag, expected, 1) == expected) {
            *found_index = idx;
        }
    }
}

typedef struct {
    int thread_id;
    int length;
    const unsigned char *target_digest;
    u64 batch_size;

    /* Per-thread GPU resources -- own stream, own device buffer, own
     * device found-flag/index. No two threads ever share any of these,
     * so no lock or atomic is needed for GPU coordination. */
    cudaStream_t stream;
    char *d_batch_buf;
    int *d_found_flag;
    int *d_found_index;
    char *host_batch;

    /* outputs */
    u64 nodes_expanded, leaf_tests, gpu_batches_launched;
    double astar_elapsed, gpu_time;
    int found_here;
    char found_password[MAX_LEN + 1];
} ThreadArg;

/* Dynamic task queue: TOTAL_TASKS = g_charset_len^2 independent
 * 2-character root prefixes, claimed one at a time via atomic fetch-add. */
static std::atomic<int> g_next_task(0);
static struct timespec g_search_start;

/* Flushes this thread's own local batch to its own stream/buffer. Returns
 * 1 if a match was found in this batch. */
static int flush_batch(ThreadArg *arg, int batch_count, int cand_len) {
    if (batch_count == 0) return 0;

    CUDA_CHECK(cudaMemcpyAsync(arg->d_batch_buf, arg->host_batch,
                                (size_t)batch_count * (MAX_LEN + 1),
                                cudaMemcpyHostToDevice, arg->stream));

    int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(arg->d_found_flag, &zero, sizeof(int),
                                cudaMemcpyHostToDevice, arg->stream));

    int blocks = (batch_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    batch_test_kernel<<<blocks, THREADS_PER_BLOCK, 0, arg->stream>>>(
        arg->d_batch_buf, batch_count, cand_len, arg->d_found_flag, arg->d_found_index);

    int found_flag_host;
    CUDA_CHECK(cudaMemcpyAsync(&found_flag_host, arg->d_found_flag, sizeof(int),
                                cudaMemcpyDeviceToHost, arg->stream));
    /* Synchronize stream before checking result */
    CUDA_CHECK(cudaStreamSynchronize(arg->stream));

    if (found_flag_host) {
        int found_idx_host;
        CUDA_CHECK(cudaMemcpy(&found_idx_host, arg->d_found_index, sizeof(int), cudaMemcpyDeviceToHost));
        memcpy(arg->found_password, arg->host_batch + (size_t)found_idx_host * (MAX_LEN + 1), (size_t)cand_len);
        arg->found_password[cand_len] = '\0';
        return 1;
    }
    return 0;
}

static void *worker(void *arg_ptr) {
    ThreadArg *arg = (ThreadArg *)arg_ptr;
    int length = arg->length;

    /* Per-thread CUDA setup -- own stream and device resources, created
     * once at thread start, destroyed once at thread end. */
    CUDA_CHECK(cudaStreamCreate(&arg->stream));
    CUDA_CHECK(cudaMalloc(&arg->d_batch_buf, arg->batch_size * (MAX_LEN + 1)));
    CUDA_CHECK(cudaMalloc(&arg->d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&arg->d_found_index, sizeof(int)));
    /* Pinned host memory for fast DMA transfers */
    CUDA_CHECK(cudaHostAlloc((void **)&arg->host_batch, arg->batch_size * (MAX_LEN + 1), cudaHostAllocDefault));
    u64 batch_count = 0;

    Heap heap;
    heap_init(&heap, 1024);
    int root_depth = (length >= 2) ? 2 : 1;
    int total_tasks = (length >= 2) ? (g_charset_len * g_charset_len) : g_charset_len;

    u64 nodes_expanded = 0, leaf_tests = 0, gpu_batches = 0;
    double gpu_time_total = 0.0;
    int found_here = 0;

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (;;) {
        int task_id = g_next_task.fetch_add(1);
        if (task_id >= total_tasks) break; /* no tasks left to claim */
        if (g_found_flag.load() || found_here) break;

        /* Reuse the same heap object across tasks -- reset size to 0 but
         * keep whatever capacity it already grew to. */
        heap.size = 0;

        State root;
        root.depth = root_depth;
        root.g = (double)root_depth * EDGE_COST;
        if (root_depth == 2) {
            int first = task_id / g_charset_len;
            int second = task_id % g_charset_len;
            root.prefix[0] = g_charset[first];
            root.prefix[1] = g_charset[second];
            root.prefix[2] = '\0';
        } else {
            root.prefix[0] = g_charset[task_id];
            root.prefix[1] = '\0';
        }
        root.f = root.g + HEURISTIC_WEIGHT * (length - root_depth) * EDGE_COST;
        heap_push(&heap, root);

    while (heap.size > 0) {
        if (g_found_flag.load()) break;

        State s = heap_pop(&heap);

        if (s.depth == length) {
            leaf_tests++;
            memcpy(arg->host_batch + batch_count * (MAX_LEN + 1), s.prefix, (size_t)length + 1);
            batch_count++;

            if (batch_count == arg->batch_size) {
                struct timespec g0, g1;
                clock_gettime(CLOCK_MONOTONIC, &g0);
                int hit = flush_batch(arg, (int)batch_count, length);
                clock_gettime(CLOCK_MONOTONIC, &g1);
                gpu_time_total += (g1.tv_sec - g0.tv_sec) + (g1.tv_nsec - g0.tv_nsec) / 1e9;
                gpu_batches++;
                batch_count = 0;
                if (hit) {
                    int expected = 0;
                    if (g_found_flag.compare_exchange_strong(expected, 1)) {
                        pthread_mutex_lock(&g_result_mutex);
                        strcpy(g_found_password, arg->found_password);
                        pthread_mutex_unlock(&g_result_mutex);
                    }
                    found_here = 1;
                    break;
                }
            }
            continue;
        }

        nodes_expanded++;
        for (int c = 0; c < g_charset_len; c++) {
            State child;
            child.depth = s.depth + 1;
            memcpy(child.prefix, s.prefix, (size_t)s.depth);
            child.prefix[s.depth] = g_charset[c];
            child.prefix[s.depth + 1] = '\0';
            child.g = s.g + EDGE_COST;
            child.f = child.g + HEURISTIC_WEIGHT * (length - child.depth) * EDGE_COST;
            heap_push(&heap, child);
        }

        heap_trim_if_needed(&heap);

        if (nodes_expanded % 5000 == 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - g_search_start.tv_sec) + (now.tv_nsec - g_search_start.tv_nsec) / 1e9;
            if (elapsed >= TIME_BUDGET_SECONDS) { found_here = -1; break; }
            if (g_found_flag.load()) break;
        }
        if (nodes_expanded >= NODE_BUDGET) { found_here = -1; break; }
    }
    if (found_here == -1) { found_here = 0; break; } /* budget exceeded: stop claiming new tasks too */
    if (found_here == 1) break; /* success: stop claiming new tasks immediately */
    /* Otherwise: this task's search space genuinely exhausted (heap
     * emptied) -- loop back to the top and claim the next available task. */
    }

    /* Flush any remaining partial batch -- don't leave accumulated
     * leaves untested just because this thread's search ended. */
    if (!found_here && !g_found_flag.load() && batch_count > 0) {
        struct timespec g0, g1;
        clock_gettime(CLOCK_MONOTONIC, &g0);
        int hit = flush_batch(arg, (int)batch_count, length);
        clock_gettime(CLOCK_MONOTONIC, &g1);
        gpu_time_total += (g1.tv_sec - g0.tv_sec) + (g1.tv_nsec - g0.tv_nsec) / 1e9;
        gpu_batches++;
        if (hit) {
            int expected = 0;
            if (g_found_flag.compare_exchange_strong(expected, 1)) {
                pthread_mutex_lock(&g_result_mutex);
                strcpy(g_found_password, arg->found_password);
                pthread_mutex_unlock(&g_result_mutex);
            }
            found_here = 1;
        }
    }

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    arg->astar_elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    arg->nodes_expanded = nodes_expanded;
    arg->leaf_tests = leaf_tests;
    arg->gpu_batches_launched = gpu_batches;
    arg->gpu_time = gpu_time_total;
    arg->found_here = found_here;

    heap_free(&heap);
    cudaFreeHost(arg->host_batch);
    cudaFree(arg->d_batch_buf);
    cudaFree(arg->d_found_flag);
    cudaFree(arg->d_found_index);
    cudaStreamDestroy(arg->stream);

    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <length> <target_password> [--threads N] [--batch-size N]\n", argv[0]);
        return 1;
    }

    int length = atoi(argv[1]);
    const char *target_password = argv[2];
    int num_threads = 0;
    u64 batch_size = DEFAULT_BATCH_SIZE;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
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

    if (num_threads <= 0) {
        long detected = sysconf(_SC_NPROCESSORS_ONLN);
        num_threads = (detected > 0) ? (int)detected : 1;
    }
    if (num_threads < 1) num_threads = 1;

    unsigned char target_digest[16];
    md5((const unsigned char *)target_password, (size_t)length, target_digest);
    char target_hex[33];
    for (int i = 0; i < 16; i++) sprintf(target_hex + i * 2, "%02x", target_digest[i]);
    target_hex[32] = '\0';

    CUDA_CHECK(cudaMemcpyToSymbol(d_target_digest, target_digest, 16));

    int total_tasks_display = (length >= 2) ? (g_charset_len * g_charset_len) : g_charset_len;
    printf("Password len  : %d\n", length);
    printf("Charset       : %d symbols (fixed, not selectable)\n", g_charset_len);
    printf("Cost model    : uniform\n");
    printf("Target MD5    : %s\n", target_hex);
    printf("Threads       : %d (dynamic %d-task queue, each with its own CUDA stream)\n", num_threads, total_tasks_display);
    printf("GPU batch/thr : %llu\n", (unsigned long long)batch_size);
    printf("============================================================\n");
    fflush(stdout);

    ThreadArg *args = (ThreadArg *)calloc((size_t)num_threads, sizeof(ThreadArg));
    pthread_t *tids = (pthread_t *)calloc((size_t)num_threads, sizeof(pthread_t));

    for (int t = 0; t < num_threads; t++) {
        args[t].thread_id = t;
        args[t].length = length;
        args[t].target_digest = target_digest;
        args[t].batch_size = batch_size;
    }

    struct timespec prog_t0, prog_t1;
    clock_gettime(CLOCK_MONOTONIC, &prog_t0);
    g_search_start = prog_t0; /* shared across all threads for the global time budget */

    for (int t = 0; t < num_threads; t++) {
        pthread_create(&tids[t], NULL, worker, &args[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(tids[t], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &prog_t1);
    double total_time = (prog_t1.tv_sec - prog_t0.tv_sec) + (prog_t1.tv_nsec - prog_t0.tv_nsec) / 1e9;

    u64 total_nodes = 0, total_leaves = 0, total_gpu_batches = 0;
    double total_gpu_time = 0.0;
    for (int t = 0; t < num_threads; t++) {
        total_nodes += args[t].nodes_expanded;
        total_leaves += args[t].leaf_tests;
        total_gpu_batches += args[t].gpu_batches_launched;
        total_gpu_time += args[t].gpu_time;
    }

    printf("============================================================\n");
    if (g_found_flag.load()) {
        printf("CRACKED       : %s\n", g_found_password);
    } else {
        printf("Password NOT recovered.\n");
    }
    printf("Total nodes expanded (all threads)     : %llu\n", total_nodes);
    printf("Total leaf tests (all threads)         : %llu\n", total_leaves);
    printf("Total GPU batches (all threads/streams) : %llu\n", total_gpu_batches);
    printf("Total GPU time (summed across threads) : %.3fs\n", total_gpu_time);
    printf("Elapsed time   : %.6f seconds\n", total_time);
    printf("============================================================\n");

    free(args);
    free(tids);
    return 0;
}
