#include <stdlib.h>
#include <stdio.h>
#include "astar_heap.h"

void heap_init(Heap *h, int initial_capacity) {
    h->data = (State *)malloc(sizeof(State) * (size_t)initial_capacity);
    if (h->data == NULL) {
        fprintf(stderr, "Fatal: out of memory allocating initial heap (%d states)\n", initial_capacity);
        exit(1);
    }
    h->size = 0;
    h->capacity = initial_capacity;
}

void heap_free(Heap *h) { 
    free(h->data); 
    h->data = NULL; 
}

static inline int state_better(const State *a, const State *b) {
    if (a->f < b->f) return 1;
    if (a->f == b->f && a->depth > b->depth) return 1;
    return 0;
}

void heap_push(Heap *h, State s) {
    if (h->size == h->capacity) {
        h->capacity *= 2;
        State *new_data = (State *)realloc(h->data, sizeof(State) * (size_t)h->capacity);
        if (new_data == NULL) {
            fprintf(stderr, "Fatal: out of memory growing heap to %d states\n", h->capacity);
            exit(1);
        }
        h->data = new_data;
    }
    int i = h->size++;
    h->data[i] = s;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!state_better(&h->data[i], &h->data[parent])) break;
        State tmp = h->data[parent]; h->data[parent] = h->data[i]; h->data[i] = tmp;
        i = parent;
    }
}

State heap_pop(Heap *h) {
    State top = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    for (;;) {
        int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
        if (left < h->size && state_better(&h->data[left], &h->data[smallest])) smallest = left;
        if (right < h->size && state_better(&h->data[right], &h->data[smallest])) smallest = right;
        if (smallest == i) break;
        State tmp = h->data[i]; h->data[i] = h->data[smallest]; h->data[smallest] = tmp;
        i = smallest;
    }
    return top;
}

static void swap_state(State *a, State *b) { State tmp = *a; *a = *b; *b = tmp; }

static void quickselect(State *arr, int lo, int hi, int k) {
    while (lo < hi) {
        int pivot_idx = lo + (hi - lo) / 2;
        State pivot = arr[pivot_idx];
        int i = lo, j = hi;
        while (i <= j) {
            while (state_better(&arr[i], &pivot)) i++;
            while (state_better(&pivot, &arr[j])) j--;
            if (i <= j) {
                swap_state(&arr[i], &arr[j]);
                i++;
                j--;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else return;
    }
}

static void sift_down(State *arr, int n, int i) {
    for (;;) {
        int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
        if (left < n && state_better(&arr[left], &arr[smallest])) smallest = left;
        if (right < n && state_better(&arr[right], &arr[smallest])) smallest = right;
        if (smallest == i) break;
        State tmp = arr[i]; arr[i] = arr[smallest]; arr[smallest] = tmp;
        i = smallest;
    }
}

static void heapify(State *arr, int n) { 
    for (int i = n / 2 - 1; i >= 0; i--) sift_down(arr, n, i); 
}

void heap_trim_if_needed(Heap *h) {
    if (h->size <= BEAM_CAP) return;
    quickselect(h->data, 0, h->size - 1, BEAM_KEEP - 1);
    h->size = BEAM_KEEP;
    heapify(h->data, h->size);
}
