import torch
import torch.nn as nn

from bitpack import unpack32
import cvxq_cuda

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking CVXQ mixed-precision matvec ...')

DEV = torch.device('cuda')

M = 12288 # * 4 # * 4
N = 12288 * 4
D = [7,1,3,3,1,7,2,3,5,1,2,5,1,2,7,1,1,7,6,2,4,2,1,4,3,5,4,1,2,2,1,0] # mean is exactly 3
# D = [3]

DTYPE = torch.float
mat = torch.randint(-1000000000, 1000000000, (M // 32 * 3, N), device=DEV, dtype=torch.int)
vec = torch.randn((1, M), device=DEV, dtype=torch.half).to(DTYPE)
mul = torch.zeros((1, N), device=DEV, dtype=torch.half).to(DTYPE)

depths = torch.tensor(D, device=DEV, dtype=torch.uint8).repeat(M // len(D) // 4)[torch.randperm(M // 4)].contiguous()
scales = torch.randn(M // 4, device=DEV, dtype=DTYPE) #torch.half).to(DTYPE)

cumsum = (depths.repeat_interleave(4).cumsum(0) - depths.repeat_interleave(4)).contiguous()
i_s = ((cumsum * N // 32)[torch.arange(0, M, 256).reshape(-1,1)] + torch.arange(0, N, 256, device=DEV)).to(torch.int)
shifts = ((cumsum % 32)[torch.arange(0, M, 256).reshape(-1,1)]).repeat([1, N//256]).to(torch.uint8)

COUNT = 1000
import time
# run once to get the cvxq kernel loaded into memory
cvxq_cuda.vecquant3matmul(vec, mat, mul, depths, scales, i_s, shifts)
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    cvxq_cuda.vecquant3matmul(vec, mat, mul, depths, scales, i_s, shifts)
torch.cuda.synchronize()
time_cvxq = (time.time() - tick) / COUNT
print('CVXQ:', time_cvxq)

DTYPE = torch.half
mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
# mat = unpack32(mat, depths).to(DTYPE)
# mat = ((mat + 0.5) * (1 / 2 ** depths.reshape(-1,1)) * scales.reshape(-1,1) + zeros.reshape(-1,1)).to(DTYPE)

vec = vec.to(DTYPE)
mul = mul.to(DTYPE)

zeros = torch.randn(M, device=DEV, dtype=torch.half).to(DTYPE);

COUNT = 1000
import time
# run once to get the cublas kernel loaded into memory
torch.matmul(vec, mat, out=mul)
torch.cuda.synchronize()

tick = time.time()
for _ in range(COUNT):
    torch.matmul(vec, mat, out=mul)
torch.cuda.synchronize()
time_fp16 = (time.time() - tick) / COUNT
print('FP16:', time_fp16)

print('Speed-up:', time_fp16 / time_cvxq)
