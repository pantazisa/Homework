/*
 * crack_astar.c
 * Sequential baseline implementation of A* password recovery.
 */

#include "astar_heap.h"
#include "md5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CHARSET_LEN 128



typedef unsigned long long u64;

static char g_charset[MAX_CHARSET_LEN];
static int g_charset_len;
static int g_char_to_idx[256];

/* 
 * Uniform cost per character transition:
 * Note: Under uniform cost and h = length - depth, f(n) = g(n) + h(n) = length is
 * constant across nodes. The priority queue acts as a deeper-first tie-broken frontier
 * steering exploration directly toward leaves, while beam trimming acts as a protective
 * memory safety bound.
 */
static const double EDGE_COST = 1.0;

static int solve(int length, const unsigned char *target_digest,
                 char *found_password) {
  Heap heap;
  heap_init(&heap, 1024);

  State start;
  start.depth = 0;
  start.g = 0.0;
  start.f = length * HEURISTIC_WEIGHT * EDGE_COST;
  start.prefix[0] = '\0';
  heap_push(&heap, start);

  u64 nodes_expanded = 0, leaf_tests = 0;
  int found = 0;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  while (heap.size > 0) {
    State s = heap_pop(&heap);

    if (s.depth == length) {
      leaf_tests++;
      unsigned char digest[16];
      md5((const unsigned char *)s.prefix, (size_t)length, digest);
      if (md5_equal(digest, target_digest)) {
        found = 1;
        strcpy(found_password, s.prefix);
        break;
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

    if (nodes_expanded % TIME_CHECK_INTERVAL == 0) {
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      double elapsed_so_far =
          (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
      if (elapsed_so_far >= TIME_BUDGET_SECONDS) {
        printf("    A* time budget (%.0fs) reached -- stopping.\n",
               TIME_BUDGET_SECONDS);
        break;
      }
    }
    if (nodes_expanded >= NODE_BUDGET) {
      printf("    A* node budget reached -- stopping.\n");
      break;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double astar_elapsed =
      (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  heap_free(&heap);

  printf("    A*: %llu nodes expanded, %llu leaf tests, %.3fs\n",
         nodes_expanded, leaf_tests, astar_elapsed);

  return found;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <length> <target_password>\n", argv[0]);
    return 1;
  }

  int length = atoi(argv[1]);
  const char *target_password = argv[2];

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

  unsigned char target_digest[16];
  md5((const unsigned char *)target_password, (size_t)length, target_digest);
  char target_hex[33];
  md5_to_hex(target_digest, target_hex);

  printf("Password len  : %d\n", length);
  printf("Charset       : %d symbols (fixed, not selectable)\n", g_charset_len);
  printf("Cost model    : uniform\n");
  printf("Target MD5    : %s\n", target_hex);
  printf("============================================================\n");
  fflush(stdout);

  char found_password[MAX_LEN + 1] = {0};

  struct timespec prog_t0, prog_t1;
  clock_gettime(CLOCK_MONOTONIC, &prog_t0);

  int found = solve(length, target_digest, found_password);

  clock_gettime(CLOCK_MONOTONIC, &prog_t1);
  double total_time = (prog_t1.tv_sec - prog_t0.tv_sec) +
                      (prog_t1.tv_nsec - prog_t0.tv_nsec) / 1e9;

  printf("============================================================\n");
  if (found) {
    printf("CRACKED       : %s\n", found_password);
  } else {
    printf("Password NOT recovered.\n");
  }
  printf("Elapsed time   : %.6f seconds\n", total_time);
  printf("============================================================\n");

  return 0;
}