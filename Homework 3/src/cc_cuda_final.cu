#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

typedef struct Graph {
    int vertices;
    long long num_edges;
    int *edges;
    long long *offsets;
    int *labels;
} Graph;

// --- 1. FIXED FILE IO ---

inline int fast_parse_int(char *&p) {
    int val = 0;
    while (*p && (*p < '0' || *p > '9')) p++; 
    if (!*p) return -1;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return val;
}

// Helper to skip weights/remaining text on a line
inline void skip_line(char *&p) {
    while (*p && *p != '\n') p++;
    if (*p == '\n') p++;
}

Graph* createGraph(int vertices) {
    Graph* g = (Graph*)malloc(sizeof(Graph));
    if (!g) return NULL;
    g->vertices = vertices;
    g->num_edges = 0;
    g->edges = NULL;
    g->offsets = (long long*)calloc((size_t)vertices + 1, sizeof(long long));
    g->labels = (int*)malloc((size_t)vertices * sizeof(int));
    for (int i = 0; i < vertices; i++) g->labels[i] = i; 
    return g;
}

Graph *readMTX_Fast(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) return NULL;

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    char *map = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { close(fd); return NULL; }

    char *p = map;
    while (*p == '%') { skip_line(p); }

    int rows = fast_parse_int(p);
    int cols = fast_parse_int(p);
    long long nnz = fast_parse_int(p);
    skip_line(p);

    int n = (rows > cols) ? rows : cols;
    Graph *g = createGraph(n);
    int *temp = (int*)calloc(n, sizeof(int));

    char *p_pass1 = p;
    for (long long i = 0; i < nnz; i++) {
        int u = fast_parse_int(p_pass1) - 1;
        int v = fast_parse_int(p_pass1) - 1;
        skip_line(p_pass1); // Skip the weight column
        if (u >= 0 && v >= 0 && u < n && v < n && u != v) {
            temp[u]++; temp[v]++;
        }
    }

    g->offsets[0] = 0;
    for (int i = 1; i <= n; i++) g->offsets[i] = g->offsets[i-1] + temp[i-1];
    g->num_edges = g->offsets[n];
    g->edges = (int*)malloc((size_t)g->num_edges * sizeof(int));
    
    for (int i = 0; i < n; i++) temp[i] = 0;

    char *p_pass2 = p;
    for (long long i = 0; i < nnz; i++) {
        int u = fast_parse_int(p_pass2) - 1;
        int v = fast_parse_int(p_pass2) - 1;
        skip_line(p_pass2); // Skip the weight column
        if (u >= 0 && v >= 0 && u < n && v < n && u != v) {
            g->edges[g->offsets[u] + (size_t)temp[u]++] = v;
            g->edges[g->offsets[v] + (size_t)temp[v]++] = u;
        }
    }

    munmap(map, file_size); close(fd); free(temp);
    return g;
}

Graph* loadBinGraph(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    int n; long long num_edges;
    if (fread(&n, sizeof(int), 1, f) != 1 || fread(&num_edges, sizeof(long long), 1, f) != 1) { fclose(f); return NULL; }
    Graph* g = createGraph(n);
    g->num_edges = num_edges;
    g->edges = (int*)malloc((size_t)num_edges * sizeof(int));
    size_t r1 = fread(g->offsets, sizeof(long long), n+1, f);
    size_t r2 = fread(g->edges, sizeof(int), num_edges, f);
    (void)r1; (void)r2; fclose(f); return g;
}

void saveBinGraph(Graph* g, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&g->vertices, sizeof(int), 1, f);
    fwrite(&g->num_edges, sizeof(long long), 1, f);
    fwrite(g->offsets, sizeof(long long), g->vertices + 1, f);
    fwrite(g->edges, sizeof(int), g->num_edges, f);
    fclose(f);
}

void freeGraph(Graph* g) {
    if (!g) return;
    free(g->edges); free(g->offsets); free(g->labels); free(g);
}

// --- 2. CUDA KERNELS ---

__global__ void init_labels_kernel(int *labels, int n) {
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) labels[v] = v;
}

__global__ void cc_sampling_kernel(const long long *offsets, const int *edges, int *labels, int n) {
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        long long start = offsets[v];
        long long end = offsets[v+1];
        int my_root = labels[v];
        for (long long k = start; k < end && k < start + 2; k++) {
            int neighbor = edges[k];
            if (neighbor < my_root) my_root = neighbor;
        }
        if (my_root < labels[v]) atomicMin(&labels[v], my_root);
    }
}

__global__ void cc_link_kernel(const long long *offsets, const int *edges, int *labels, bool *changed, int n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t warp_id = tid / 32;
    int lane_id = tid % 32;
    if (warp_id < (size_t)n) {
        int v = (int)warp_id;
        long long start = offsets[v];
        long long end = offsets[v+1];
        int p_v = labels[v];
        int min_label = p_v;
        for (long long k = start + lane_id; k < end; k += 32) {
            int neighbor_label = labels[edges[k]];
            if (neighbor_label < min_label) min_label = neighbor_label;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            int remote = __shfl_down_sync(0xFFFFFFFF, min_label, offset);
            if (remote < min_label) min_label = remote;
        }
        min_label = __shfl_sync(0xFFFFFFFF, min_label, 0);
        if (lane_id == 0 && min_label < p_v) {
            atomicMin(&labels[v], min_label);
            atomicMin(&labels[p_v], min_label);
            *changed = true;
        }
    }
}

__global__ void cc_compress_kernel(int *labels, int n) {
    size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int p = labels[v];
        int pp = labels[p];
        if (p != pp) labels[v] = pp;
    }
}

void ColoringAlgorithmCUDA(Graph* g) {
    int n = g->vertices;
    int *d_labels, *d_edges; long long *d_offsets; bool *d_changed, h_changed;
    cudaCheck(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));
    cudaCheck(cudaMalloc(&d_offsets, (size_t)(n + 1) * sizeof(long long)));
    cudaCheck(cudaMalloc(&d_edges, (size_t)g->num_edges * sizeof(int)));
    cudaCheck(cudaMalloc(&d_changed, sizeof(bool)));
    cudaCheck(cudaMemcpy(d_offsets, g->offsets, (size_t)(n + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_edges, g->edges, (size_t)g->num_edges * sizeof(int), cudaMemcpyHostToDevice));
    int threads = 256;
    size_t blocks_v = ((size_t)n + threads - 1) / threads;
    size_t blocks_warp = ((size_t)n * 32 + threads - 1) / threads;
    init_labels_kernel<<<blocks_v, threads>>>(d_labels, n);
    cc_sampling_kernel<<<blocks_v, threads>>>(d_offsets, d_edges, d_labels, n);
    cc_compress_kernel<<<blocks_v, threads>>>(d_labels, n);
    int iterations = 0;
    do {
        h_changed = false;
        cudaCheck(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
        cc_link_kernel<<<blocks_warp, threads>>>(d_offsets, d_edges, d_labels, d_changed, n);
        cc_compress_kernel<<<blocks_v, threads>>>(d_labels, n);
        cudaDeviceSynchronize();
        cudaCheck(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        iterations++;
    } while (h_changed && iterations < 50);
    for(int i=0; i<4; i++) cc_compress_kernel<<<blocks_v, threads>>>(d_labels, n);
    printf("Converged in %d iterations.\n", iterations);
    cudaCheck(cudaMemcpy(g->labels, d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_labels); cudaFree(d_offsets); cudaFree(d_edges); cudaFree(d_changed);
}

// --- 3. MAIN (ORIGINAL PRINTS RETAINED) ---

int main(int argc, char* argv[]) {
    if (argc < 2) { printf("Usage: %s <file.mtx>\n", argv[0]); return 1; }
    char bin_name[256]; snprintf(bin_name, sizeof(bin_name), "%s.bin", argv[1]);
    
    printf("Loading graph...\n");
    Graph* g = loadBinGraph(bin_name);
    if (!g) {
        g = readMTX_Fast(argv[1]);
        if (!g) return 1;
        saveBinGraph(g, bin_name);
    } else printf("Loaded binary: %s\n", bin_name);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    ColoringAlgorithmCUDA(g);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    int num_components = 0;
    for (int i = 0; i < g->vertices; i++) if (g->labels[i] == i) num_components++;

    printf("Total Vertices: %d\nComponents: %d\nGPU Kernel Time: %f s\n", g->vertices, num_components, time_taken);
    freeGraph(g); return 0;
}