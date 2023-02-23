#include <cstdint>
#include <cuda.h>
#include <tuple>

namespace arithmetic {
//TODO(Alan WANG): UT needed.
void __device__ __forceinline__ mac(uint64_t a, uint64_t b, uint64_t c,
                                    uint64_t carry, uint64_t *ho,
                                    uint64_t *lo) {
  __uint128_t ret =
      (__uint128_t)a + ((__uint128_t)b * (__uint128_t)c) + (__uint128_t)carry;
  *lo = (uint64_t)ret;
  *ho = ret >> 64;
}

//TODO(Alan WANG): UT needed.
void __device__ __forceinline__ adc(uint64_t a, uint64_t b, uint64_t carry, uint64_t *ho, uint64_t *lo) {
  __uint128_t ret = (__uint128_t)a + (__uint128_t)b + (__uint128_t)carry;
  *lo = (uint64_t)ret;
  *ho = ret >> 64;
}

//TODO(Alan WANG): UT needed.
void __device__ __forceinline__ sbb(uint64_t a, uint64_t b, uint64_t borrow, uint64_t *ho, uint64_t *lo) {
  __uint128_t ret = (__uint128_t)a - ((__uint128_t)b + (__uint128_t)(borrow >> 63));
  *lo = (uint64_t)ret;
  *ho = ret >> 64;
}
}; // namespace arithmetic

namespace ff {
struct FF {
  uint64_t bytes[4];
};

/// q = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
constexpr FF MODULUS = {{
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
}};

/// R = 2^256 mod q
constexpr FF R = {{
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
}};

/// R^2 = 2^512 mod q
constexpr FF R2 = {{
    0xf32cfc5b538afa89,
    0xb5e71911d44501fb,
    0x47ab1eff0a417ff6,
    0x06d89f71cab8351f,
}};

/// R^3 = 2^768 mod q
constexpr FF R3 = {{
    0xb1cd6dafda1530df,
    0x62f210e6a7283db6,
    0xef7f0b0c0ada0afb,
    0x20fd6e902d592544,
}};

/// = -1 mod q
constexpr FF NEGATIVE_ONE = {{
    0x68c3488912edefaa,
    0x8d087f6872aabf4f,
    0x51e1a24709081231,
    0x2259d6b14729c0fa,
}};

constexpr uint64_t INV = 0x87d20782e4866389;

bool __device__ __forceinline__ is_zero(FF *v) {
  uint64_t res = 0xffffffffffffffff;
  for (auto i = 0; i < 4; i++) {
    res = res & v->bytes[i];
  }
  return res == 0;
}

FF square(FF *v) {
  // TODO(Alan WANG)
  return *v;
}

// len(r) == 8
void __device__ __forceinline__ montgomery_reduce(uint64_t *r, FF* ret) {
  uint64_t k, carry0, carry1, borrow, unused, r0, r1, r2, r3, r4, r5, r6, r7;
  k = r[0] * INV;
  arithmetic::mac(r[0], k, MODULUS.bytes[0],      0, &r0, &carry0);
  arithmetic::mac(r[1], k, MODULUS.bytes[1], carry0, &r1, &carry0);
  arithmetic::mac(r[2], k, MODULUS.bytes[2], carry0, &r2, &carry0);
  arithmetic::mac(r[3], k, MODULUS.bytes[3], carry0, &r3, &carry0);
  arithmetic::adc(r[4], 0, carry0, &r4, &carry1);

  k = r1 * INV;
  arithmetic::mac(r1, k, MODULUS.bytes[0],      0, &r1, &carry0);
  arithmetic::mac(r2, k, MODULUS.bytes[1], carry0, &r2, &carry0);
  arithmetic::mac(r3, k, MODULUS.bytes[2], carry0, &r3, &carry0);
  arithmetic::mac(r4, k, MODULUS.bytes[3], carry0, &r4, &carry0);
  arithmetic::adc(r[5], carry1, carry0, &r5, &carry1);

  k = r2 * INV;
  arithmetic::mac(r2, k, MODULUS.bytes[0],      0, &r2, &carry0);
  arithmetic::mac(r3, k, MODULUS.bytes[1], carry0, &r3, &carry0);
  arithmetic::mac(r4, k, MODULUS.bytes[2], carry0, &r4, &carry0);
  arithmetic::mac(r5, k, MODULUS.bytes[3], carry0, &r5, &carry0);
  arithmetic::adc(r[6], carry1, carry0, &r6, &carry1);

  k = r3 * INV;
  arithmetic::mac(r3, k, MODULUS.bytes[0],      0, &r3, &carry0);
  arithmetic::mac(r4, k, MODULUS.bytes[1], carry0, &r4, &carry0);
  arithmetic::mac(r5, k, MODULUS.bytes[2], carry0, &r5, &carry0);
  arithmetic::mac(r6, k, MODULUS.bytes[3], carry0, &r6, &carry0);
  arithmetic::adc(r[7], carry1, carry0, &r7, &carry1);

  arithmetic::sbb(r4, MODULUS.bytes[0],      0, &(ret->bytes[0]), &borrow);
  arithmetic::sbb(r5, MODULUS.bytes[1], borrow, &(ret->bytes[1]), &borrow);
  arithmetic::sbb(r6, MODULUS.bytes[2], borrow, &(ret->bytes[2]), &borrow);
  arithmetic::sbb(r7, MODULUS.bytes[3], borrow, &(ret->bytes[3]), &borrow);
  arithmetic::sbb(carry1, 0, borrow, &unused, &borrow);

  arithmetic::adc(ret->bytes[0], MODULUS.bytes[0] & borrow,      0, &(ret->bytes[0]), &carry0);
  arithmetic::adc(ret->bytes[1], MODULUS.bytes[1] & borrow, carry0, &(ret->bytes[1]), &carry0);
  arithmetic::adc(ret->bytes[2], MODULUS.bytes[2] & borrow, carry0, &(ret->bytes[2]), &carry0);
  arithmetic::adc(ret->bytes[3], MODULUS.bytes[3] & borrow, carry0, &(ret->bytes[3]), &carry0);
}

} // namespace ff

struct G1 {
  ff::FF x;
  ff::FF y;
  ff::FF z;
};

struct G1Affine {
  ff::FF x;
  ff::FF y;
};

namespace bn256 {
__device__ bool is_identity(G1 *v) { return ff::is_zero(&(v->z)); }

__device__ void add(G1 *lhs, G1 *rhs, G1 *out) {
  if (is_identity(lhs)) {
    *out = *rhs;
    return;
  }

  if (is_identity(rhs)) {
    *out = *lhs;
    return;
  }
}
} // namespace bn256

extern "C" __global__ void ff_cpy(void *src, void *dst, unsigned int len) {
  memcpy(dst, src, len * sizeof(G1Affine));
}
