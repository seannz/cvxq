import torch
import modules

weight = torch.tensor(loadmat('weights04.mat')['weight'])[:,5:6]
quantizer = modules.Quantizer()

bit_depths = torch.arange(0, 9)
step_sizes = torch.arange(-16,16,0.125)
dist1 = torch.zeros(len(bit_depths), len(step_sizes))
dist2 = torch.zeros(len(bit_depths), len(step_sizes))

for j in range(len(bit_depths)):
    base_size = torch.round(torch.log2(3 * weight.reshape(-1).std() / (2 ** bit_depths[j])))
    for i in range(len(step_sizes)):
        quantizer.quantize(bit_depths[j], base_size + step_sizes[i])
        dist1[j, i] = torch.mean((quantizer(weight) - weight) ** 2)

for j in range(len(bit_depths)):
    base_size = torch.round(torch.log2(1 / (2 ** bit_depths[j])))
    for i in range(len(step_sizes)):
        quantizer.quantize(bit_depths[j], base_size + step_sizes[i], compand=True)
        dist2[j, i] = torch.mean((quantizer(weight) - weight) ** 2)

