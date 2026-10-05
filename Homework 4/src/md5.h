#ifndef MD5_H
#define MD5_H

#include <stddef.h>
#include <stdint.h>

/*
 * Minimal, self-contained single-block MD5 implementation (RFC 1321).
 *
 * Kept deliberately simple and dependency-free (no OpenSSL) for fast,
 * portable CPU verification of candidate passwords up to 55 bytes.
 *
 * digest must point to a 16-byte buffer.
 */
#ifdef __cplusplus
extern "C" {
#endif

void md5(const unsigned char *initial_msg, size_t initial_len,
         unsigned char *digest);

/* Convenience: compare two 16-byte digests. Returns 1 if equal, 0 otherwise. */
int md5_equal(const unsigned char *a, const unsigned char *b);

/* Convenience: print a 16-byte digest as 32 hex chars into a 33-byte buffer. */
void md5_to_hex(const unsigned char *digest, char *out_hex33);

#ifdef __cplusplus
}
#endif

#endif /* MD5_H */
