#include <cuda.h>
extern "C" __constant__ int my_constant = 314;
extern "C" __global__ void make_buckets(const uint32_t *bases,
                                        const uint32_t *scalars,
                                        uint32_t *buckets, int buf_len) {
  auto tid = threadIdx.x;
  for (auto i = 0; i < buf_len; i++) {
    if (tid == scalars[i]) {
      buckets[tid] += bases[i];
    }
  }
}
