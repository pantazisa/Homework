/*
 * crack_astar_omp.c
 * OpenMP parallel implementation of A* password recovery.
 *
 * Uses dynamic 2-character prefix task scheduling with thread-local min-heaps.
 */

#include "astar_heap.h"
#include "md5.h"
#include <omp.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_CHARSET_LEN 128



typedef unsigned long long u64;

static char g_charset[MAX_CHARSET_LEN];
static int g_charset_len;
static int g_char_to_idx[256];

static atomic_int g_found_flag = 0;
static char g_found_password[MAX_LEN + 1];

/* Uniform cost per character transition */
static const double EDGE_COST = 1.0;

typedef struct {
  int length;
  const unsigned char *target_digest;
  u64 nodes_expanded, leaf_tests;
} ThreadArg;

/* Per-thread heaps, indexed by omp_get_thread_num(), reused across
 * however many dynamically-scheduled tasks a given thread ends up
 * processing -- avoids repeated malloc/free between tasks. Allocated
 * once before the parallel region, sized to num_threads. */
static Heap *g_thread_heaps;
static struct timespec g_search_start;
static atomic_ullong g_total_nodes_expanded = 0;

/* Processes a single root prefix task. */
static void process_task(ThreadArg *arg, int task_id, int root_depth) {
  int length = arg->length;
  int tid = omp_get_thread_num();
  Heap *heap = &g_thread_heaps[tid];
  heap->size = 0;

  State s2;
  s2.depth = root_depth;
  s2.g = (double)root_depth * EDGE_COST;
  if (root_depth == 2) {
    int first = task_id / g_charset_len;
    int second = task_id % g_charset_len;
    s2.prefix[0] = g_charset[first];
    s2.prefix[1] = g_charset[second];
    s2.prefix[2] = '\0';
  } else {
    s2.prefix[0] = g_charset[task_id];
    s2.prefix[1] = '\0';
  }
  s2.f = s2.g + HEURISTIC_WEIGHT * (length - root_depth) * EDGE_COST;
  heap_push(heap, s2);

  while (heap->size > 0) {
    if (atomic_load(&g_found_flag))
      return;

    State s = heap_pop(heap);

    if (s.depth == length) {
      arg->leaf_tests++;
      unsigned char digest[16];
      md5((const unsigned char *)s.prefix, (size_t)length, digest);
      if (md5_equal(digest, arg->target_digest)) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&g_found_flag, &expected, 1)) {
          strcpy(g_found_password, s.prefix);
        }
        return;
      }
      continue;
    }

    arg->nodes_expanded++;

    for (int c = 0; c < g_charset_len; c++) {
      State child;
      child.depth = s.depth + 1;
      memcpy(child.prefix, s.prefix, (size_t)s.depth);
      child.prefix[s.depth] = g_charset[c];
      child.prefix[s.depth + 1] = '\0';
      child.g = s.g + EDGE_COST;
      child.f = child.g + HEURISTIC_WEIGHT * (length - child.depth) * EDGE_COST;
      heap_push(heap, child);
    }

    heap_trim_if_needed(heap);

    if (arg->nodes_expanded % TIME_CHECK_INTERVAL == 0) {
      u64 total_now = atomic_fetch_add(&g_total_nodes_expanded, TIME_CHECK_INTERVAL) + TIME_CHECK_INTERVAL;
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      double elapsed_so_far = (now.tv_sec - g_search_start.tv_sec) +
                              (now.tv_nsec - g_search_start.tv_nsec) / 1e9;
      if (elapsed_so_far >= TIME_BUDGET_SECONDS)
        return;
      if (total_now >= NODE_BUDGET)
        return;
      if (atomic_load(&g_found_flag))
        return;
    }
  }
  /* Task's search space genuinely exhausted -- OpenMP will hand this
   * thread the next available task_id automatically. */
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <length> <target_password> [--threads N]\n",
            argv[0]);
    return 1;
  }

  int length = atoi(argv[1]);
  const char *target_password = argv[2];
  int num_threads = 0;

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      num_threads = atoi(argv[++i]);
    }
  }

  if (length <= 0 || length > MAX_LEN) {
    fprintf(stderr, "Error: length must be between 1 and %d\n", MAX_LEN);
    return 1;
  }
  if ((int)strlen(target_password) != length) {
    fprintf(stderr, "Error: target password length must equal <length>\n");
    return 1;
  }

  strcpy(g_charset, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
                    "6789!@#$%^&*()-_=+");
  g_charset_len = (int)strlen(g_charset);
  for (int i = 0; i < 256; i++)
    g_char_to_idx[i] = -1;
  for (int i = 0; i < g_charset_len; i++)
    g_char_to_idx[(unsigned char)g_charset[i]] = i;

  for (int i = 0; i < length; i++) {
    if (g_char_to_idx[(unsigned char)target_password[i]] < 0) {
      fprintf(stderr, "Error: target password contains a character outside the "
                      "supported alphabet\n");
      return 1;
    }
  }

  if (num_threads <= 0) {
    long detected = sysconf(_SC_NPROCESSORS_ONLN);
    num_threads = (detected > 0) ? (int)detected : 1;
  }
  if (num_threads < 1)
    num_threads = 1;

  unsigned char target_digest[16];
  md5((const unsigned char *)target_password, (size_t)length, target_digest);
  char target_hex[33];
  md5_to_hex(target_digest, target_hex);

  int root_depth = (length >= 2) ? 2 : 1;
  int total_tasks =
      (length >= 2) ? (g_charset_len * g_charset_len) : g_charset_len;

  printf("Password len  : %d\n", length);
  printf("Charset       : %d symbols (fixed, not selectable)\n", g_charset_len);
  printf("Cost model    : uniform\n");
  printf("Target MD5    : %s\n", target_hex);
  printf("Threads       : %d\n", num_threads);
  printf("Design        : dynamic 2-character task queue (%d tasks, "
         "schedule(dynamic,1))\n",
         total_tasks);
  printf("============================================================\n");
  fflush(stdout);

  ThreadArg *args = calloc((size_t)num_threads, sizeof(ThreadArg));
  g_thread_heaps = calloc((size_t)num_threads, sizeof(Heap));
  for (int t = 0; t < num_threads; t++) {
    args[t].length = length;
    args[t].target_digest = target_digest;
    heap_init(&g_thread_heaps[t], 1024);
  }

  struct timespec prog_t0, prog_t1;
  clock_gettime(CLOCK_MONOTONIC, &prog_t0);
  g_search_start = prog_t0;

  omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
  for (int task_id = 0; task_id < total_tasks; task_id++) {
    if (!atomic_load(&g_found_flag)) {
      process_task(&args[omp_get_thread_num()], task_id, root_depth);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &prog_t1);
  double total_time = (prog_t1.tv_sec - prog_t0.tv_sec) +
                      (prog_t1.tv_nsec - prog_t0.tv_nsec) / 1e9;

  for (int t = 0; t < num_threads; t++)
    heap_free(&g_thread_heaps[t]);
  free(g_thread_heaps);

  u64 total_nodes = 0, total_leaves = 0;
  for (int t = 0; t < num_threads; t++) {
    total_nodes += args[t].nodes_expanded;
    total_leaves += args[t].leaf_tests;
  }

  printf("============================================================\n");
  if (atomic_load(&g_found_flag)) {
    printf("CRACKED       : %s\n", g_found_password);
  } else {
    printf("Password NOT recovered.\n");
  }
  printf("Total nodes expanded (all threads) : %llu\n", total_nodes);
  printf("Total leaf tests (all threads)     : %llu\n", total_leaves);
  printf("Elapsed time   : %.6f seconds\n", total_time);
  printf("============================================================\n");

  free(args);
  return 0;
}