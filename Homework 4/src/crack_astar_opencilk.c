/*
 * crack_astar_opencilk.c
 * OpenCilk parallel implementation of A* password recovery using cilk_for.
 */

#include "astar_heap.h"
#include "md5.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
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



/* Uniform cost per character transition */
static const double EDGE_COST = 1.0;

/* Global atomic flags and counters */
static atomic_int g_found_flag = 0;
static char g_found_password[MAX_LEN + 1];
static atomic_ullong g_total_nodes = 0;
static atomic_ullong g_total_leaves = 0;

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
      fprintf(stderr, "Error: target password contains a character outside the supported alphabet\n");
      return 1;
    }
  }

  if (num_threads <= 0) {
    const char *cilk_workers_env = getenv("CILK_NWORKERS");
    if (cilk_workers_env != NULL) {
      int from_env = atoi(cilk_workers_env);
      if (from_env > 0)
        num_threads = from_env;
    }
    if (num_threads <= 0) {
      long detected = sysconf(_SC_NPROCESSORS_ONLN);
      num_threads = (detected > 0) ? (int)detected : 1;
    }
  }
  if (num_threads < 1)
    num_threads = 1;

  if (num_threads > 0) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", num_threads);
    setenv("CILK_NWORKERS", buf, 1);
  }

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
  printf("Threads       : %d (CILK_NWORKERS)\n", num_threads);
  printf("Design        : dynamic %d-task queue via cilk_for (Independent "
         "Local Heaps)\n",
         total_tasks);
  printf("============================================================\n");
  fflush(stdout);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  /*
   * Dynamically schedule prefix sub-tasks across available Cilk workers.
   * Each task uses an independent local heap to avoid synchronization overhead.
   */
  cilk_for (int task_id = 0; task_id < total_tasks; task_id++) {
    if (atomic_load(&g_found_flag))
      continue;

    Heap heap;
    heap_init(&heap, 1024);

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

    u64 local_nodes = 0;
    u64 local_leaves = 0;

    while (heap.size > 0) {
      if (atomic_load(&g_found_flag))
        break;

      State s = heap_pop(&heap);

      if (s.depth == length) {
        local_leaves++;
        unsigned char digest[16];
        md5((const unsigned char *)s.prefix, (size_t)length, digest);
        if (md5_equal(digest, target_digest)) {
          int expected = 0;
          if (atomic_compare_exchange_strong(&g_found_flag, &expected, 1)) {
            strcpy(g_found_password, s.prefix);
          }
          break;
        }
        continue;
      }

      local_nodes++;

      for (int c = 0; c < g_charset_len; c++) {
        State child;
        child.depth = s.depth + 1;
        memcpy(child.prefix, s.prefix, (size_t)s.depth);
        child.prefix[s.depth] = g_charset[c];
        child.prefix[s.depth + 1] = '\0';
        child.g = s.g + EDGE_COST;
        child.f =
            child.g + HEURISTIC_WEIGHT * (length - child.depth) * EDGE_COST;
        heap_push(&heap, child);
      }

      heap_trim_if_needed(&heap);

      if (local_nodes % TIME_CHECK_INTERVAL == 0) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed =
            (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (elapsed >= TIME_BUDGET_SECONDS)
          break;
        if (atomic_load(&g_total_nodes) + local_nodes >= NODE_BUDGET)
          break;
      }
    }

    /* Accumulate per-task metrics into global statistics */
    atomic_fetch_add(&g_total_nodes, local_nodes);
    atomic_fetch_add(&g_total_leaves, local_leaves);

    heap_free(&heap);
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double total_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

  printf("    A*: %llu nodes expanded, %llu leaf tests\n",
         (unsigned long long)atomic_load(&g_total_nodes),
         (unsigned long long)atomic_load(&g_total_leaves));
  printf("============================================================\n");
  if (atomic_load(&g_found_flag)) {
    printf("CRACKED       : %s\n", g_found_password);
  } else {
    printf("Password NOT recovered.\n");
  }
  printf("Elapsed time   : %.6f seconds\n", total_time);
  printf("============================================================\n");

  return 0;
}