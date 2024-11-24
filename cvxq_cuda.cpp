#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor depths, torch::Tensor scales, torch::Tensor i_s, torch::Tensor shifts
);

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor depths, torch::Tensor scales, torch::Tensor i_s, torch::Tensor shifts
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, depths, scales, i_s, shifts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
}
