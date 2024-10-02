import torch
import modules

# weight = torch.distributions.Laplace(0,1).sample([32 * 4096, 1]) #
weight = torch.tensor(loadmat('weights04.mat')['weight'])[:,4]
# quantizer = modules.Quantizer()

bit_depth = 3  #torch.arange(4, 5)
# step_size = -bit_depth #torch.arange(-bit_depth, -bit_depth + 1) #, 0.125)

scales = torch.arange(1,8,0.5)
step_sizes = torch.arange(-4 - bit_depth, 0 - bit_depth + 1, 0.5)
dist1 = torch.zeros(len(scales),len(step_sizes))
dist2 = torch.zeros(len(scales),len(step_sizes))

for i, scale in enumerate(scales):
    for j, step_size in enumerate(step_sizes):
        quants = torch.distributions.Laplace(0,scale * weight.abs().mean()).cdf(weight) - 0.5
        # quants = torch.tanh(0.02 * weight) #- 0.5
        quants = torch.floor(quants * (2. ** -step_size))
        quants = torch.clamp(quants, -(2. ** (bit_depth - 1)), (2. ** (bit_depth -1)) - 1) + 0.5
        # figure(); histogram(quants.reshape(-1),bins=torch.arange(-(2.**(bit_depth - 1)),(2.**(bit_depth - 1))))
        quants = quants * (2. ** step_size)
        quants = torch.distributions.Laplace(0, scale * weight.abs().mean()).icdf(quants + 0.5)
        # quants = torch.atanh(quants) / 0.02
        dist1[i,j] = 0.5 * torch.log2(torch.mean((quants - weight) ** 2))
        print(scale, step_size, dist1[i,j])


for i, scale in enumerate(scales):
    for j, step_size in enumerate(step_sizes):
        quants = torch.distributions.Normal(0,scale * weight.std()).cdf(weight) - 0.5
        # quants = torch.tanh(0.02 * weight) #- 0.5
        quants = torch.floor(quants * (2. ** -step_size))
        quants = torch.clamp(quants, -(2. ** (bit_depth - 1)), (2. ** (bit_depth -1)) - 1) + 0.5
        # figure(); histogram(quants.reshape(-1),bins=torch.arange(-(2.**(bit_depth - 1)),(2.**(bit_depth - 1))))
        quants = quants * (2. ** step_size)
        quants = torch.distributions.Normal(0, scale * weight.std()).icdf(quants + 0.5)
        # quants = torch.atanh(quants) / 0.02
        dist2[i,j] = 0.5 * torch.log2(torch.mean((quants - weight) ** 2))
        print(scale, step_size, dist1[i,j])



# figure(); histogram(quants.reshape(-1),bins=torch.arange(-(2.**(bit_depth - 1)),(2.**(bit_depth - 1))) * (2 ** -bit_depth))

for step_size in torch.arange(-12, 6, 0.25):
    quants = weight # torch.distributions.Laplace(0,1).cdf(weight) - 0.5
    quants = torch.floor(quants * (2. ** -step_size))
    quants = torch.clamp(quants, -(2. ** (bit_depth - 1)), (2. ** (bit_depth -1)) - 1) + 0.5
    # figure(); histogram(quants.reshape(-1),bins=torch.arange(-(2.**(bit_depth - 1)),(2.**(bit_depth - 1))))
    quants = quants * (2. ** step_size)
    dist2 = 0.5 * torch.log2(torch.mean((quants - weight) ** 2))
    print(dist2)
# figure(); histogram(quants.reshape(-1),bins=torch.arange(-(2.**(bit_depth - 1)),(2.**(bit_depth - 1))) * (2 ** step_size))





# dist1 = torch.zeros(len(step_sizes))
# base_size = torch.round(torch.log2(3 * weight.reshape(-1).std() / (2 ** bit_depth)))
# step_sizes = torch.arange(-8,8,0.25)
# for i in range(len(step_sizes)):
#     quantizer.quantize(bit_depth, base_size + step_sizes[i])
#     dist1[i] = torch.mean((quantizer(weight) - weight) ** 2)


# step_sizes = torch.arange(-8.5,-8.375,0.125)
# dist1 = torch.zeros(len(bit_depths), len(step_sizes))
# dist2 = torch.zeros(len(bit_depths), len(step_sizes))



# for j in range(len(bit_depths)):
#     base_size = torch.round(torch.log2(3 * weight.reshape(-1).std() / (2 ** bit_depths[j])))
#     for i in range(len(step_sizes)):
#         quantizer.quantize(bit_depths[j], base_size + step_sizes[i])
#         dist1[j, i] = torch.mean((quantizer(weight) - weight) ** 2)

# for j in range(len(bit_depths)):
#     base_size = 1 / (2 ** bit_depths[j])
#     for i in range(len(step_sizes)):
#         quantizer.quantize(bit_depths[j], base_size + step_sizes[i], compand=True)
#         dist2[j, i] = torch.mean((quantizer(weight) - weight) ** 2)

