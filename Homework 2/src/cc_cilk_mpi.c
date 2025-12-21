#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <cilk/cilk.h>
#include <time.h>
#include <mpi.h>
#include <cilk/cilk_api.h>
#include <string.h>

typedef struct Graph {
    int vertices;
    long long num_edges;
    int *edges;
    long long *offsets;
    int *labels;
} Graph;

void* safe_malloc(size_t size, const char* name, int rank) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "[Rank %d] FATAL: Failed to allocate %.2f GB for %s\n", 
                rank, (double)size/1e9, name);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

Graph* createGraph(int vertices, int rank) {
    Graph* g = safe_malloc(sizeof(Graph), "Graph Struct", rank);
    g->vertices = vertices;
    g->num_edges = 0;
    g->offsets = calloc((size_t)vertices + 1, sizeof(long long));
    g->labels = safe_malloc((size_t)vertices * sizeof(int), "Labels", rank);
    g->edges = NULL;

    cilk_for(int i = 0; i < vertices; i++) {
        g->labels[i] = i; 
    }
    return g;
}

void freeGraph(Graph* g) {
    if (!g) return;
    free(g->edges);
    free(g->offsets);
    free(g->labels);
    free(g);
}

void BroadcastGraph(Graph** g_ptr, int rank) {
    int n = 0;
    long long num_edges = 0;
    if (rank == 0) {
        if (*g_ptr) { n = (*g_ptr)->vertices; num_edges = (*g_ptr)->num_edges; }
        else { n = -1; }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n <= 0) { MPI_Finalize(); exit(1); }
    MPI_Bcast(&num_edges, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        *g_ptr = createGraph(n, rank);
        (*g_ptr)->num_edges = num_edges;
        (*g_ptr)->edges = safe_malloc((size_t)num_edges * sizeof(int), "Edges", rank);
    }
    MPI_Bcast((*g_ptr)->offsets, n + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    long long offset = 0;
    int chunk = 500000000; 
    while (offset < num_edges) {
        int to_send = (num_edges - offset > chunk) ? chunk : (int)(num_edges - offset);
        MPI_Bcast(&((*g_ptr)->edges[offset]), to_send, MPI_INT, 0, MPI_COMM_WORLD);
        offset += to_send;
    }
}

Graph *readMTX(const char* filename, int rank) {
    FILE* f = fopen(filename, "r");
    if (!f) return NULL;
    char line[1024];
    while (fgets(line, sizeof(line), f) && (line[0] == '%' || line[0] == '#'));

    int rows, cols;
    long long nnz;
    if (sscanf(line, "%d %d %lld", &rows, &cols, &nnz) != 3) { fclose(f); return NULL; }
    int n = (rows > cols) ? rows : cols;

    Graph *g = createGraph(n, rank);
    long long *temp_count = calloc(n, sizeof(long long)); // Use 64-bit counts
    
    // Pass 1: Count degrees (Handles SuiteSparse weights automatically)
    int u, v;
    for (long long i = 0; i < nnz; i++) {
        if (fscanf(f, "%d %d%*[^\n]", &u, &v) == 2) {
            u--; v--; // 1-based to 0-based
            if (u >= 0 && v >= 0 && u < n && v < n && u != v) {
                temp_count[u]++; temp_count[v]++;
            }
        }
    }
    g->offsets[0] = 0;
    for (int i = 0; i < n; i++) g->offsets[i+1] = g->offsets[i] + temp_count[i];
    g->num_edges = g->offsets[n];
    g->edges = safe_malloc((size_t)g->num_edges * sizeof(int), "Edges", rank);
    
    // Pass 2: Fill edges
    memset(temp_count, 0, (size_t)n * sizeof(long long));
    rewind(f);
    while (fgets(line, sizeof(line), f) && (line[0] == '%' || line[0] == '#'));
    fgets(line, sizeof(line), f); // skip header
    for (long long i = 0; i < nnz; i++) {
        if (fscanf(f, "%d %d%*[^\n]", &u, &v) == 2) {
            u--; v--;
            if (u >= 0 && v >= 0 && u < n && v < n && u != v) {
                g->edges[g->offsets[u] + temp_count[u]++] = v;
                g->edges[g->offsets[v] + temp_count[v]++] = u;
            }
        }
    }
    free(temp_count); fclose(f);
    return g;
}

void ColoringAlgorithmHybrid(Graph* g, int rank, int size) {
    int n = g->vertices;
    int chunk = n / size;
    int start_v = rank * chunk;
    int end_v = (rank == size - 1) ? n : (rank + 1) * chunk;

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int r_start = i * chunk;
        int r_end = (i == size - 1) ? n : (i + 1) * chunk;
        recvcounts[i] = r_end - r_start;
        displs[i] = r_start;
    }

    int global_changed = 1;
    while (global_changed) {
        int local_changed = 0;
        cilk_for(int v = start_v; v < end_v; v++) {
            for (long long k = g->offsets[v]; k < g->offsets[v+1]; k++) {
                int u = g->edges[k];
                if (g->labels[v] > g->labels[u]) {
                    g->labels[v] = g->labels[u];
                    local_changed = 1; 
                }
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, g->labels, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    free(recvcounts); free(displs);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <file.mtx>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    Graph* g = NULL;
    if (rank == 0) {
        printf("[Rank 0] Loading %s...\n", argv[1]);
        g = readMTX(argv[1], rank);
    }

    BroadcastGraph(&g, rank);
    if (rank == 0) printf("Graph loaded: %d nodes, %lld entries.\n", g->vertices, g->num_edges);

    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec start, end;
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &start); 

    ColoringAlgorithmHybrid(g, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end); 
        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        int comps = 0;
        for (int i = 0; i < g->vertices; i++) if (g->labels[i] == i) comps++;
        printf("Nodes: %d | Components: %d | Time: %f s\n", g->vertices, comps, time);
    }

    freeGraph(g);
    MPI_Finalize();
    return 0;
}