#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

 
typedef struct Graph{ //CSR Graph struct
    int vertices;
    long long num_edges;
    int *edges;
    long long *offsets;
    int *labels;
}Graph;

Graph * createGraph(int vertices){ // Create empty graph
    
    Graph* g = malloc(sizeof(Graph));
    
    if(!g){
        return NULL;
    } 
    g->vertices = vertices;
    g->num_edges = 0;
    g->edges = NULL;
    g->offsets = calloc(vertices + 1, sizeof(long long));
    g->labels = malloc(vertices * sizeof(int));
    
    for(int i = 0; i < vertices; i++){
        g->labels[i] = i; 
    }
    return g;
}

void freeGraph(Graph* g){ // Free memory allocated for the graph
    
    if(!g){
        return;
    }

    free(g->edges);
    free(g->offsets);
    free(g->labels);
    free(g);
}
void saveBinGraph(Graph* g, const char* filename) { // Save graph to binary file the first time the program is run
    
    FILE* f = fopen(filename, "wb");
    
    if (!f){
        return;
    }
    fwrite(&g->vertices, sizeof(int), 1, f);
    fwrite(&g->num_edges, sizeof(long long), 1, f);
    fwrite(g->offsets, sizeof(long long), g->vertices + 1, f);
    fwrite(g->edges, sizeof(int), g->num_edges, f);
    fclose(f);
    printf("Saved binary file: %s\n", filename);
}

Graph* loadBinGraph(const char* filename) { // Load graph from binary file after having been saved
    FILE* f = fopen(filename, "rb");
    
    if (!f) {
        return NULL;
    }

    int n;
    long long num_edges;
    
    if (fread(&n, sizeof(int), 1, f) != 1) {
        fclose(f); 
        return NULL;
    }
    
    if (fread(&num_edges, sizeof(long long), 1, f) != 1) { 
        fclose(f);
        return NULL; 
    }

    Graph* g = createGraph(n);

    g->num_edges = num_edges;
    g->edges = malloc(num_edges * sizeof(int));
    
    fread(g->offsets, sizeof(long long), n + 1, f);
    fread(g->edges, sizeof(int), num_edges, f);
    
    fclose(f);
    return g;
}

Graph *readMTX(const char* filename){ // Read graph from .mtx file
    FILE* f = fopen(filename, "r");

    if(!f){
        return NULL;
    }

    char line[1024]; // to skip comments in mtx files
    
    while(fgets(line, sizeof(line), f)){
        
        if(line[0] != '%'){
            break;
        }
    }

    int rows , cols;
    long long nnz;
    
    if(sscanf(line, "%d %d %lld", &rows, &cols, &nnz) != 3){
        fclose(f);
        return NULL;
    }
    
    int n = (rows > cols) ? rows : cols;
    
    Graph *g = createGraph(n);
    long data_start = ftell(f);
    int *temp = calloc(n, sizeof(int));
    
    if(!temp){
        fclose(f);
        return NULL;
    }
     
    long long count = 0;
    
    while(count < nnz && fgets(line, sizeof(line), f)){
        int u , v;
        if(sscanf(line, "%d %d", &u, &v) == 2){
            u--; v--; 
            if (u >= n || v >= n){
                continue;
            }
            if (u == v){
                continue;
            }
            temp[u]++;
            temp[v]++;            
            count++;
        }
    }
    
    g->offsets[0] = 0;
    
    for(int i =1; i <= n; i++){
        g->offsets[i] = g->offsets[i-1] + temp[i-1];
    }
    
    g->num_edges = g->offsets[n];
    g->edges = malloc(g->num_edges * sizeof(int));
    
    if(!g->edges){
        printf("NOT ENOUGH MEMORY\n");
        free(temp);
        fclose(f);
        return NULL;
    } 
    rewind(f);
    fseek(f, data_start, SEEK_SET); 
    
    for(int i= 0; i <n; i++){
        temp[i] = 0;
    }

    count = 0;

    while(count < nnz && fgets(line, sizeof(line), f)){
        
        int u, v;
        
        if(sscanf(line, "%d %d", &u, &v) == 2){
            u--;
            v--; 
            if (u >= n || v >= n){
                continue;
            } 
            if (u == v) {
                continue;
            }
            
            int u1 = g->offsets[u] + temp[u]++; // position to insert v in u's adjacency list
            g->edges[u1] = v;
            
            int v1 = g->offsets[v] + temp[v]++; // position to insert u in v's adjacency list
            g->edges[v1] = u;
            
            count++;
        }
    }
    free(temp);
    fclose(f);
    return g;
}
void ColoringAlgorithm(Graph* g){

    int n = g->vertices;
    int *labels = g->labels;
    long long *offsets = g->offsets;
    int *edges = g->edges;
    
    for(int i=0;i<n;i++){
        labels[i] = i;
    }
    
    bool changed = true;
    
    while(changed){

        changed = false;

        for(int v=0;v<n;v++){
            
            long long start = g->offsets[v];
            long long end = g->offsets[v+1];

            for(long long k=start; k<end; k++){

                int u = g->edges[k];

                if(g->labels[v] > g->labels[u]){
                    g->labels[v] = g->labels[u];
                    changed = true;
                }
            }
        }
    }
}

    

int main(int argc, char* argv[]){
    
    if(argc < 2){
        printf("opening: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }
    
    char bin_name[256];
    snprintf(bin_name, sizeof(bin_name), "%s.bin", argv[1]);
    Graph* g = loadBinGraph(bin_name);
    
    if(!g){
        g = readMTX(argv[1]);
        
        if(!g){
            printf("Failed to load graph from %s\n", argv[1]);
            return 1;
        }
        saveBinGraph(g, bin_name);
    }
    else{
        printf("Loaded binary graph: %s\n", bin_name);
    }
    
    clock_t start_time = clock(); // Start Timer
    ColoringAlgorithm(g);
    clock_t end_time = clock(); // End Timer
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    int num_components = 0;
    for(int i=0; i<g->vertices; i++){
        if(g->labels[i] == i){
            num_components++;
        }
    }
    printf("Total Vertices: %d\n", g->vertices);
    printf("Number of Connected Components: %d\n", num_components);
    printf("time taken: %f seconds\n", time_taken);
    freeGraph(g);
    return 0;
}