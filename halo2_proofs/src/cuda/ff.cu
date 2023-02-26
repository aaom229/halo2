#include <cstdint>
#include <cuda.h>

namespace ff {
struct FF {
  uint64_t bytes[4];
};
struct FFRepr {
  uint8_t bytes[32];
};
void __device__ __forceinline__ montgomery_reduce(uint64_t *, FF *);

} // namespace ff

namespace arithmetic {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 6
void __device__ __forceinline__ mac(uint64_t a, uint64_t b, uint64_t c,
                                    uint64_t carry, uint64_t *lo,
                                    uint64_t *ho) {
  __uint128_t ret =
      (__uint128_t)a + ((__uint128_t)b * (__uint128_t)c) + (__uint128_t)carry;
  *lo = (uint64_t)ret;
  *ho = ret >> 64;
}

void __device__ __forceinline__ adc(uint64_t a, uint64_t b, uint64_t carry,
                                    uint64_t *lo, uint64_t *ho) {
  __uint128_t ret = (__uint128_t)a + (__uint128_t)b + (__uint128_t)carry;
  *lo = (uint64_t)ret;
  *ho = ret >> 64;
}

void __device__ __forceinline__ sbb(uint64_t a, uint64_t b, uint64_t borrow,
                                    uint64_t *lo, uint64_t *ho) {
  __uint128_t ret =
      (__uint128_t)a - ((__uint128_t)b + (__uint128_t)(borrow >> 63));
  *lo = (uint64_t)ret;
  *ho = ret >> 64;
}
#else
void __device__ __forceinline__ mac32(uint32_t a, uint32_t b, uint32_t c,
                                      uint32_t carry, uint32_t *lo,
                                      uint32_t *ho) {
  uint64_t ret = (uint64_t)a + ((uint64_t)b * (uint64_t)c) + (uint64_t)carry;
  *lo = (uint32_t)ret;
  *ho = ret >> 32;
  *ho = 0;
}

void __device__ __forceinline__ adc(uint64_t a, uint64_t b, uint64_t carry,
                                    uint64_t *lo, uint64_t *ho) {
  uint32_t r[3];
  uint64_t ret = 0ull + (uint32_t)carry + (uint32_t)a + (uint32_t)b;
  r[2] = ret >> 32;
  r[0] = (uint32_t)ret;
  ret = (a >> 32) + (b >> 32) + (carry >> 32) + r[2];
  r[2] = ret >> 32;
  r[1] = (uint32_t)ret;
  *lo = r[0] | ((uint64_t)(r[1]) << 32);
  *ho = r[2];
}

void __device__ __forceinline__ mac(uint64_t a, uint64_t b, uint64_t c,
                                    uint64_t carry, uint64_t *lo,
                                    uint64_t *ho) {
  uint32_t r[4], t0;
  mac32((uint32_t)a, (uint32_t)b, (uint32_t)c, (uint32_t)carry, &r[0], &t0);
  a = (a >> 32) + (carry >> 32);
  mac32((uint32_t)a, (uint32_t)b, (uint32_t)(c >> 32), t0, &r[1], &t0);
  mac32(r[1], (uint32_t)(b >> 32), (uint32_t)c, 0, &r[1], &r[2]);
  a = (uint64_t)t0 + (a >> 32);
  mac32(r[2], (uint32_t)(b >> 32), (uint32_t)(c >> 32), (uint32_t)a, &r[2],
        &r[3]);
  *lo = r[0] | ((uint64_t)(r[1]) << 32);
  *ho = r[2] | ((uint64_t)(r[3]) << 32);
}

void __device__ __forceinline__ sbb(uint64_t a, uint64_t b, uint64_t borrow,
                                    uint64_t *lo, uint64_t *ho) {
  uint32_t r[4];
  uint64_t ret = (uint32_t)a - ((uint32_t)b + (borrow >> 63));
  r[0] = (uint32_t)ret;
  r[2] = ret >> 32;
  ret = (a >> 32) - ((b >> 32) + (r[2] >> 31));
  r[1] = (uint32_t)ret;
  r[3] = ret >> 32;
  *lo = r[0] | ((uint64_t)(r[1]) << 32);
  *ho = r[3] | ((uint64_t)(r[3]) << 32); // It's right!
}
#endif
}; // namespace arithmetic

namespace ff {
/// q = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
constexpr __device__ FF MODULUS = {{
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

FF __device__ __forceinline__ zero() { return FF({0, 0, 0, 0}); }

uint8_t __device__ __forceinline__ get(const FF &v, uint32_t idx) {
  return ((uint8_t *)((void *)&v))[idx];
}

bool __device__ __forceinline__ is_zero(const FF &v) {
  uint64_t res = 0xffffffffffffffff;
  for (auto i = 0; i < 4; i++) {
    res = res & v.bytes[i];
  }
  return res == 0;
}

void __device__ __forceinline__ sub(const FF &lhs, const FF &rhs, FF *out) {
  uint64_t borrow = 0, carry = 0;
  arithmetic::sbb(lhs.bytes[0], rhs.bytes[0], borrow, &(out->bytes[0]),
                  &borrow);
  arithmetic::sbb(lhs.bytes[1], rhs.bytes[1], borrow, &(out->bytes[1]),
                  &borrow);
  arithmetic::sbb(lhs.bytes[2], rhs.bytes[2], borrow, &(out->bytes[2]),
                  &borrow);
  arithmetic::sbb(lhs.bytes[3], rhs.bytes[3], borrow, &(out->bytes[3]),
                  &borrow);

  arithmetic::adc(out->bytes[0], MODULUS.bytes[0] & borrow, carry,
                  &(out->bytes[0]), &carry);
  arithmetic::adc(out->bytes[1], MODULUS.bytes[1] & borrow, carry,
                  &(out->bytes[1]), &carry);
  arithmetic::adc(out->bytes[2], MODULUS.bytes[2] & borrow, carry,
                  &(out->bytes[2]), &carry);
  arithmetic::adc(out->bytes[3], MODULUS.bytes[3] & borrow, carry,
                  &(out->bytes[3]), &carry);
}

void __device__ __forceinline__ add(const FF &lhs, const FF &rhs, FF *out) {
  uint64_t carry = 0;
  arithmetic::adc(lhs.bytes[0], rhs.bytes[0], carry, &(out->bytes[0]), &carry);
  arithmetic::adc(lhs.bytes[1], rhs.bytes[1], carry, &(out->bytes[1]), &carry);
  arithmetic::adc(lhs.bytes[2], rhs.bytes[2], carry, &(out->bytes[2]), &carry);
  arithmetic::adc(lhs.bytes[3], rhs.bytes[3], carry, &(out->bytes[3]), &carry);
  sub(*out, MODULUS, out);
}

void __device__ __forceinline__ dbl(const FF &self, FF *out) {
  return add(self, self, out);
}

void __device__ __forceinline__ square(const FF &self, FF *out) {
  uint64_t r[8], carry = 0;
  arithmetic::mac(0, self.bytes[0], self.bytes[1], 0, &r[1], &carry);
  arithmetic::mac(0, self.bytes[0], self.bytes[2], carry, &r[2], &carry);
  arithmetic::mac(0, self.bytes[0], self.bytes[3], carry, &r[3], &r[4]);

  arithmetic::mac(r[3], self.bytes[1], self.bytes[2], 0, &r[3], &carry);
  arithmetic::mac(r[4], self.bytes[1], self.bytes[3], carry, &r[4], &r[5]);
  arithmetic::mac(r[5], self.bytes[2], self.bytes[3], 0, &r[5], &r[6]);

  r[7] = r[6] >> 63;
  r[6] = (r[6] << 1) | (r[5] >> 63);
  r[5] = (r[5] << 1) | (r[4] >> 63);
  r[4] = (r[4] << 1) | (r[3] >> 63);
  r[3] = (r[3] << 1) | (r[2] >> 63);
  r[2] = (r[2] << 1) | (r[1] >> 63);
  r[1] = r[1] << 1;

  arithmetic::mac(0, self.bytes[0], self.bytes[0], 0, &r[0], &carry);
  arithmetic::adc(0, r[1], carry, &r[1], &carry);
  arithmetic::mac(r[2], self.bytes[1], self.bytes[1], carry, &r[2], &carry);
  arithmetic::adc(0, r[3], carry, &r[3], &carry);
  arithmetic::mac(r[4], self.bytes[2], self.bytes[2], carry, &r[4], &carry);
  arithmetic::adc(0, r[5], carry, &r[5], &carry);
  arithmetic::mac(r[6], self.bytes[3], self.bytes[3], carry, &r[6], &carry);
  arithmetic::adc(0, r[7], carry, &r[7], &carry);

  montgomery_reduce(r, out);
}

void __device__ __forceinline__ mul(const FF &lhs, const FF &rhs, FF *out) {
  uint64_t r[8], carry = 0;
  arithmetic::mac(0, lhs.bytes[0], rhs.bytes[0], 0, &r[0], &carry);
  arithmetic::mac(0, lhs.bytes[0], rhs.bytes[1], carry, &r[1], &carry);
  arithmetic::mac(0, lhs.bytes[0], rhs.bytes[2], carry, &r[2], &carry);
  arithmetic::mac(0, lhs.bytes[0], rhs.bytes[3], carry, &r[3], &r[4]);

  arithmetic::mac(r[1], lhs.bytes[1], rhs.bytes[0], 0, &r[1], &carry);
  arithmetic::mac(r[2], lhs.bytes[1], rhs.bytes[1], carry, &r[2], &carry);
  arithmetic::mac(r[3], lhs.bytes[1], rhs.bytes[2], carry, &r[3], &carry);
  arithmetic::mac(r[4], lhs.bytes[1], rhs.bytes[3], carry, &r[4], &r[5]);

  arithmetic::mac(r[2], lhs.bytes[2], rhs.bytes[0], 0, &r[2], &carry);
  arithmetic::mac(r[3], lhs.bytes[2], rhs.bytes[1], carry, &r[3], &carry);
  arithmetic::mac(r[4], lhs.bytes[2], rhs.bytes[2], carry, &r[4], &carry);
  arithmetic::mac(r[5], lhs.bytes[2], rhs.bytes[3], carry, &r[5], &r[6]);

  arithmetic::mac(r[3], lhs.bytes[3], rhs.bytes[0], 0, &r[3], &carry);
  arithmetic::mac(r[4], lhs.bytes[3], rhs.bytes[1], carry, &r[4], &carry);
  arithmetic::mac(r[5], lhs.bytes[3], rhs.bytes[2], carry, &r[5], &carry);
  arithmetic::mac(r[6], lhs.bytes[3], rhs.bytes[3], carry, &r[6], &r[7]);

  montgomery_reduce(r, out);
}

// len(r) == 8
void __device__ __forceinline__ montgomery_reduce(uint64_t *r, FF *ret) {
  uint64_t k, carry0, carry1, borrow, unused, r0, r1, r2, r3, r4, r5, r6, r7;
  k = r[0] * INV;
  arithmetic::mac(r[0], k, MODULUS.bytes[0], 0, &r0, &carry0);
  arithmetic::mac(r[1], k, MODULUS.bytes[1], carry0, &r1, &carry0);
  arithmetic::mac(r[2], k, MODULUS.bytes[2], carry0, &r2, &carry0);
  arithmetic::mac(r[3], k, MODULUS.bytes[3], carry0, &r3, &carry0);
  arithmetic::adc(r[4], 0, carry0, &r4, &carry1);

  k = r1 * INV;
  arithmetic::mac(r1, k, MODULUS.bytes[0], 0, &r1, &carry0);
  arithmetic::mac(r2, k, MODULUS.bytes[1], carry0, &r2, &carry0);
  arithmetic::mac(r3, k, MODULUS.bytes[2], carry0, &r3, &carry0);
  arithmetic::mac(r4, k, MODULUS.bytes[3], carry0, &r4, &carry0);
  arithmetic::adc(r[5], carry1, carry0, &r5, &carry1);

  k = r2 * INV;
  arithmetic::mac(r2, k, MODULUS.bytes[0], 0, &r2, &carry0);
  arithmetic::mac(r3, k, MODULUS.bytes[1], carry0, &r3, &carry0);
  arithmetic::mac(r4, k, MODULUS.bytes[2], carry0, &r4, &carry0);
  arithmetic::mac(r5, k, MODULUS.bytes[3], carry0, &r5, &carry0);
  arithmetic::adc(r[6], carry1, carry0, &r6, &carry1);

  k = r3 * INV;
  arithmetic::mac(r3, k, MODULUS.bytes[0], 0, &r3, &carry0);
  arithmetic::mac(r4, k, MODULUS.bytes[1], carry0, &r4, &carry0);
  arithmetic::mac(r5, k, MODULUS.bytes[2], carry0, &r5, &carry0);
  arithmetic::mac(r6, k, MODULUS.bytes[3], carry0, &r6, &carry0);
  arithmetic::adc(r[7], carry1, carry0, &r7, &carry1);

  arithmetic::sbb(r4, MODULUS.bytes[0], 0, &(ret->bytes[0]), &borrow);
  arithmetic::sbb(r5, MODULUS.bytes[1], borrow, &(ret->bytes[1]), &borrow);
  arithmetic::sbb(r6, MODULUS.bytes[2], borrow, &(ret->bytes[2]), &borrow);
  arithmetic::sbb(r7, MODULUS.bytes[3], borrow, &(ret->bytes[3]), &borrow);
  arithmetic::sbb(carry1, 0, borrow, &unused, &borrow);

  arithmetic::adc(ret->bytes[0], MODULUS.bytes[0] & borrow, 0, &(ret->bytes[0]),
                  &carry0);
  arithmetic::adc(ret->bytes[1], MODULUS.bytes[1] & borrow, carry0,
                  &(ret->bytes[1]), &carry0);
  arithmetic::adc(ret->bytes[2], MODULUS.bytes[2] & borrow, carry0,
                  &(ret->bytes[2]), &carry0);
  arithmetic::adc(ret->bytes[3], MODULUS.bytes[3] & borrow, carry0,
                  &(ret->bytes[3]), &carry0);
}

bool __device__ __forceinline__ eq(const FF &lhs, const FF &rhs) {
  return ((lhs.bytes[0] ^ rhs.bytes[0]) | (lhs.bytes[1] ^ rhs.bytes[1]) |
          (lhs.bytes[2] ^ rhs.bytes[2]) | (lhs.bytes[3] ^ rhs.bytes[3])) == 0;
}

} // namespace ff

namespace bn256 {

struct G1 {
  ff::FF x;
  ff::FF y;
  ff::FF z;
};

struct G1Affine {
  ff::FF x;
  ff::FF y;
};
namespace g1 {

constexpr __device__ ff::FF G1_B = ff::FF({3, 0, 0, 0});
constexpr __device__ ff::FF G1_B_3 = ff::FF({9, 0, 0, 0});

void __device__ __forceinline__ to_identity(G1 *v) {
  v->y = ff::R;
  v->x = ff::zero();
  v->z = ff::zero();
};

bool __device__ __forceinline__ is_identity(const G1 &v) {
  return ff::is_zero(v.z);
};

void __device__ dbl(const G1 &v, G1 *out) {
  if (is_identity(v)) {
    return to_identity(out);
  }
  ff::FF a, b, c, d, e, f, x3, y3, z3;
  ff::square(v.x, &a);
  ff::square(v.y, &b);
  ff::square(b, &c);
  ff::add(v.x, b, &d);
  ff::square(d, &d);
  ff::sub(d, a, &d);
  ff::sub(d, c, &d);
  ff::dbl(d, &d);
  ff::add(a, a, &e);
  ff::add(e, a, &e);
  ff::square(e, &f);
  ff::mul(v.z, v.y, &z3);
  ff::dbl(z3, &z3);
  ff::sub(f, d, &x3);
  ff::sub(x3, d, &x3);
  ff::dbl(c, &c);
  ff::dbl(c, &c);
  ff::dbl(c, &c);
  ff::sub(d, x3, &d);
  ff::mul(e, d, &y3);
  ff::sub(y3, c, &y3);
  out->x = x3;
  out->y = y3;
  out->z = z3;
};

void __device__ add(const G1 &lhs, const G1 &rhs, G1 *out) {
  if (is_identity(lhs)) {
    *out = rhs;
    return;
  }

  if (is_identity(rhs)) {
    *out = lhs;
    return;
  }

  ff::FF z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v, x3, y3, z3;
  ff::square(lhs.z, &z1z1);
  ff::square(rhs.z, &z2z2);
  ff::mul(lhs.x, z2z2, &u1);
  ff::mul(rhs.x, z1z1, &u2);
  ff::mul(lhs.y, z2z2, &s1);
  ff::mul(s1, rhs.z, &s1);
  ff::mul(rhs.y, z1z1, &s2);
  ff::mul(s2, lhs.z, &s2);
  if (ff::eq(u1, u2)) {
    if (ff::eq(s1, s2)) {
      dbl(lhs, out);
      return;
    } else {
      to_identity(out);
      return;
    }
  }
  ff::sub(u2, u1, &h);
  ff::dbl(h, &i);
  ff::square(i, &i);
  ff::mul(h, i, &j);
  ff::sub(s2, s1, &r);
  ff::dbl(r, &r);
  ff::mul(u1, i, &v);
  ff::square(r, &x3);
  ff::sub(x3, j, &x3);
  ff::sub(x3, v, &x3);
  ff::sub(x3, v, &x3);
  ff::mul(s1, j, &s1);
  ff::dbl(s1, &s1);
  ff::sub(v, x3, &y3);
  ff::mul(r, y3, &y3);
  ff::sub(y3, s1, &y3);
  ff::add(lhs.z, rhs.z, &z3);
  ff::square(z3, &z3);
  ff::sub(z3, z1z1, &z3);
  ff::sub(z3, z2z2, &z3);
  ff::mul(z3, h, &z3);
  out->x = x3;
  out->y = y3;
  out->z = z3;
}

} // namespace g1
} // namespace bn256

namespace pippenger {
void __device__ make_buckets(bn256::G1 *bases, ff::FFRepr *scalars,
                             bn256::G1 *buckets, uint32_t buf_len,
                             uint32_t dev_idx, uint32_t segments_per_dev,
                             uint32_t blocks_per_segment) {
  uint32_t thread_idx = threadIdx.x;
  uint32_t blk_idx = blockIdx.x;
  uint32_t segment_idx =
      (dev_idx * segments_per_dev) + (blk_idx / blocks_per_segment);
  uint32_t buf_len_per_blk = buf_len / blocks_per_segment;
  uint32_t buf_offset = buf_len_per_blk * (blk_idx % blocks_per_segment);
  uint32_t bucket_offset = blk_idx << 8;

  for (uint32_t i = 0; i < buf_len_per_blk; i++) {
    uint32_t idx = i + buf_offset;
    uint8_t segment = scalars[idx].bytes[segment_idx];
    if (segment == thread_idx) {
      bn256::g1::add(buckets[bucket_offset + thread_idx], bases[idx],
                     &buckets[bucket_offset + thread_idx]);
    }
  }
}
} // namespace pippenger

extern "C" void __global__ ff_add(ff::FF *lhs, ff::FF *rhs, ff::FF *out) {
  ff::add(*lhs, *rhs, out);
}

extern "C" void __global__ ff_sub(ff::FF *lhs, ff::FF *rhs, ff::FF *out) {
  ff::sub(*lhs, *rhs, out);
}

extern "C" void __global__ ff_square(ff::FF *lhs, ff::FF *out) {
  ff::square(*lhs, out);
}

extern "C" void __global__ ff_mul(ff::FF *lhs, ff::FF *rhs, ff::FF *out) {
  ff::mul(*lhs, *rhs, out);
}

extern "C" void __global__ ff_montgomery_reduce(uint64_t *r, ff::FF *out) {
  ff::montgomery_reduce(r, out);
}

extern "C" void __global__ g1_dbl(bn256::G1 *v, bn256::G1 *out) {
  bn256::g1::dbl(*v, out);
}

extern "C" void __global__ g1_add(bn256::G1 *a, bn256::G1 *b, bn256::G1 *out) {
  bn256::g1::add(*a, *b, out);
}

extern "C" void __global__ pippenger_make_buckets(
    uint32_t *cfg, /*(buf_len, dev_idx, segments_per_dev, blocks_per_segment) */
    bn256::G1 *bases, ff::FFRepr *scalars, bn256::G1 *buckets) {
  pippenger::make_buckets(bases, scalars, buckets, cfg[0], cfg[1], cfg[2],
                          cfg[3]);
}
