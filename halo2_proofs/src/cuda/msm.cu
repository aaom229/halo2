#include <cuda.h>

#define BUCKET_LEN 256

__device__ auto make_buckets_shared(const uint32_t *bases,
                                    const uint32_t *scalars, uint32_t *buckets,
                                    int buf_len) {
  __shared__ uint32_t s_buckets[BUCKET_LEN];
  auto tid = threadIdx.x;
  for (auto i = 0; i < buf_len; i++) {
    if (tid == scalars[i]) {
      s_buckets[tid] += bases[i];
    }
  }
  __syncthreads();
  if (tid == 0) {
    memcpy(buckets, s_buckets, BUCKET_LEN * sizeof(uint32_t));
  }
  __syncthreads();
}

extern "C" __global__ void make_buckets(const uint32_t *bases,
                                        const uint32_t *scalars,
                                        uint32_t *buckets, int buf_len) {
  make_buckets_shared(bases, scalars, buckets, buf_len);
}
