#ifndef MD5_H
#define MD5_H

#include <stdint.h>
#include <stddef.h>

/*
 * Minimal, self-contained MD5 implementation (RFC 1321).
 *
 * Kept deliberately simple and dependency-free (no OpenSSL) so that the
 * exact same hashing logic can be reused verbatim inside:
 *   - the sequential CPU version (this file)
 *   - OpenMP / OpenCilk versions (still plain C, just called from parallel loops)
 *   - CUDA version (mark this function __device__ __host__ and it will
 *     compile as-is inside a .cu file with nvcc)
 *
 * digest must point to a 16-byte buffer.
 */
#ifdef __cplusplus
extern "C" {
#endif

void md5(const unsigned char *initial_msg, size_t initial_len, unsigned char *digest);

/* Convenience: compare two 16-byte digests. Returns 1 if equal, 0 otherwise. */
int md5_equal(const unsigned char *a, const unsigned char *b);

/* Convenience: print a 16-byte digest as 32 hex chars into a 33-byte buffer. */
void md5_to_hex(const unsigned char *digest, char *out_hex33);

#ifdef __cplusplus
}
#endif

#endif /* MD5_H */
