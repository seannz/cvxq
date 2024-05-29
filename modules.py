import torch
import scipy
import math
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, Identity
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import reduce
from transformers.models.opt.modeling_opt import OPTDecoderLayer 
from torch.distributions.normal import Normal


    # def gridsearch(self, layer, block, inputs, attention_masks, labels):
    #     axis = 1 # - torch.tensor(layer.linear.weight.t().shape).argmax()
    #     numchans = layer.linear.weight.shape[axis]
    #     identity = torch.eye(numchans, dtype=layer.linear.weight.dtype, device=self.model.device)
    #     channels = torch.arange(numchans).reshape([self.groupsize, max(numchans // self.groupsize, 1)])[:] # .tolist()

    #     losses = torch.nan * torch.ones(len(channels), len(self.bit_depths), len(self.step_sizes), 3, device=self.device)
    #     for c, channel in tqdm(list(enumerate(channels)), position=0):
    #         for b, bit_depth in reversed(list(enumerate(self.bit_depths))):
    #             base_size = torch.floor(torch.log2((2. ** -torch.tensor(bit_depth, device=self.model.device))))

    #             if not self.compand:
    #                 base_size += torch.floor(torch.log2(6 * layer.linear.weight.movedim(axis,0)[channel].square(e).mean().sqrt()))

    #             for s, step_size in enumerate(self.step_sizes):
    #                 # breakpoint()
    #                 qtable = 1 / identity[:,channel].sum(1, keepdims=True).movedim(0, axis) - 1 + bit_depth
    #                 layer.quantize(qtable, base_size, gain=step_size, compand=self.compand)

    #                 if step_size == 0 and bit_depth == 0:
    #                     dists = self.distortion(block, inputs, attention_masks, labels)
    #                 else:
    #                     dists = torch.nan # * torch.ones(self.num_layers, device=self.model.device)

    #                 # losses[c, b, s, 0] = step_size # base_size
    #                 losses[c, b, s, 1] = dists #self.distortion().item()
    #                 losses[c, b, s, 2] = layer.mse()

    #                 layer.unquantize()

    #     return losses

    # def distortion(self, block, inputs, attention_masks, labels):
    #     losses = torch.zeros([self.data.shape[0]], dtype=torch.float32, device=self.model.device)
    #     with torch.no_grad():
    #         # breakpoint()
    #         for b in range(0, inputs.shape[0], self.batchsize): #DataLoader(list(zip(self.data, self.logits)), batch_size=self.batchsize):
    #             output = block(inputs[b:b+self.batchsize], attention_mask=attention_masks[b:b+self.batchsize]) #.logits[:, :-1]
    #             # output = self.model(self.data[b:b+self.batchsize]) #.logits[:, :-1]
    #             losses[b:b+self.batchsize] = mse_loss(output.detach(), labels[b:b+self.batchsize], reduction='none').mean([1,2])

    #     return losses.mean()

        # forward call to measure the true channel variance (optional)
        # for i in range(len(self.model.layers)): # ,6)) + list(range(3,len(self.model.layers),6)):
        #     layer = self.model.layers[i]
        #     block = self.model
        #     labels = self.labels[-1]
        #     inputs = self.data
        #     attention_masks = None

        #     print('Optimizing ' + layer[0])
        #     self.losses = self.losses + [self.gridsearch(layer[1], block, inputs, attention_masks, labels).detach().cpu()]

        # savemat("bdcurves/%s_curves_%d_log_binormal_12b_12.mat" % (self.model_id, self.data.shape[0]), 
        #                  {"data": self.losses, "name": [self.model.layers[i][0] for i in range(len(self.model.layers))]})



        
