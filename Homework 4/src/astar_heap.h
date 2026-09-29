#ifndef ASTAR_HEAP_H
#define ASTAR_HEAP_H

#define MAX_LEN 16
#define BEAM_CAP    3000000
#define BEAM_KEEP   1200000

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char prefix[MAX_LEN + 1];
    int depth;
    double g;
    double f;
} State;

typedef struct {
    State *data;
    int size;
    int capacity;
} Heap;

void heap_init(Heap *h, int initial_capacity);
void heap_free(Heap *h);
void heap_push(Heap *h, State s);
State heap_pop(Heap *h);
void heap_trim_if_needed(Heap *h);

#ifdef __cplusplus
}
#endif

#endif // ASTAR_HEAP_H
