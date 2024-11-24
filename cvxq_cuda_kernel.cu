#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "macros.h"

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const   uint8_t* __restrict__ depths,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ i_s,
    const   uint8_t* __restrict__ shifts,
    int height,
    int width
);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;

void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor depths,
  torch::Tensor scales,
  torch::Tensor    i_s,
  torch::Tensor shifts
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        depths.data<uint8_t>(), scales.data<scalar_t>(), 
	        i_s.data<int>(), shifts.data<uint8_t>(),
        height, width
      );
    })
  );
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__constant__ float lutable[256] = { DEQUANT };

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const   uint8_t* __restrict__ depths,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ i_s,
    const   uint8_t* __restrict__ shifts,
    
    int height,
    int width) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  __shared__ scalar_t lut[BLOCKWIDTH];

  blockvec[threadIdx.x] = scales[threadIdx.x / 4] * vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = i_s[blockIdx.x * gridDim.y + blockIdx.y] + threadIdx.x;
  // int i = width * row + col;
  int shift = shifts[blockIdx.x * gridDim.y + blockIdx.y];

  uint64_t tmp_curr;
  uint32_t tmp_read;
  uint32_t depth_;

  int j = 0, k = 0;

  tmp_read = reinterpret_cast<const uint32_t*>(mat)[i];
  tmp_curr = static_cast<uint64_t>(tmp_read) << 32;
  shift += 32;
  i += width;
  // printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
  // int p = 0;

  while (k < BLOCKWIDTH) {
    depth_ = reinterpret_cast<const uint32_t*>(depths)[j];

    int depth, bmask;
    uint32_t index;
    scalar_t szero, *table;
    for (int d = 0; d < 32; d += 8) { // for each of the 4 depth clusters (represented in 8 bits)
      depth = (depth_ >> (d + 0)) &  7;
      bmask = (1 << depth) - 1;

      szero = (static_cast<int>((depth_ >> (d + 3)) & 31) - 16) * 0.03125f;
      table = reinterpret_cast<scalar_t*>(lut + (1 << depth)); // - (1 << depth);

      if (shift + 4 * depth > 64) { // will run out of bits, read more
	tmp_read = reinterpret_cast<const uint32_t*>(mat)[i];
	tmp_curr = static_cast<uint64_t>(tmp_read) << 32 | static_cast<uint64_t>(tmp_curr) >> 32;
	shift -= 32;
	i += width;
      }
      //
      index = (static_cast<uint32_t>(tmp_curr >> shift) & bmask);
      res += blockvec[k + 0] * (szero + table[index]);
      shift += depth;
      //
      index = (static_cast<uint32_t>(tmp_curr >> shift) & bmask);
      res += blockvec[k + 1] * (szero + table[index]);
      shift += depth;
      //
      index = (static_cast<uint32_t>(tmp_curr >> shift) & bmask);
      res += blockvec[k + 2] * (szero + table[index]);
      shift += depth;
      //
      index = (static_cast<uint32_t>(tmp_curr >> shift) & bmask);
      res += blockvec[k + 3] * (szero + table[index]);
      shift += depth;

      k += 4;
    }
    j += 1;
  }
  atomicAdd(&mul[col], res);
}
