import math
import torch
from torch.nn import Module
from torch.nn.functional import mse_loss, cross_entropy
from tqdm import tqdm

class Optimizer(Module):
    def __init__(self, model, data, labels, batches, batchsize=1, stride=1, length=384, loglambda=-18, bitrates=[3,4], model_id="unknown", load_file=None, save_file=None) -> None:
        super().__init__()
        self.model = model
        self.model_id = "_".join(model_id.split("/"))
        self.loglambda = loglambda
        self.bitrates = bitrates
        self.data = data
        self.labels = labels
        self.batches = batches
        self.batchsize = batchsize
        self.stride = stride
        self.length = length
        self.load_file = load_file
        self.save_file = save_file

        self.project()

        if self.load_file is None: #
            return
        
        self.load_vars()

    def calibrate(self):
        # backward call to populate the grad squared
        self.model.add_hooks()
        for b in range(0, self.batches, self.batchsize):
            # self.unquantize_all()
            for t in tqdm(list(reversed(range(0,self.data.shape[-2],self.stride)))):
                output = torch.einsum('ij,bkj->bki', self.V, self.model(self.data[b:b+self.batchsize,:t+1]).logits)
                for s in tqdm(range(output.shape[-1]), leave=False):
                    output[:,t,s].sum().backward(retain_graph = True)

            # self.quantize_all(self.loglambda, b + self.batchsize)
            for rate in self.bitrates:
                self.optimize(rate)
                self.validate()

            self.optimize(4.0)

            if self.save_file is None:
                continue
            
            self.save_vars()

        self.model.remove_hooks()

    def optimize(self, bitrate=None, log2_init=-18, lr=0.8):
        if bitrate is None:
            bitrate = self.bitrates[0]

        log2_lambda = log2_init
        self.quantize_all(log2_lambda)
        bitrate_curr = self.model.bitrate()

        iters = 0
        while abs(bitrate_curr - bitrate) > 1e-6 and iters < 10:
            log2_lambda -= 2 * (bitrate - bitrate_curr) * lr
            self.quantize_all(log2_lambda)
            bitrate_curr = self.model.bitrate()
            iters += 1

        return log2_lambda, bitrate_curr

    def project(self):
        output = torch.empty(self.data.shape, device=self.data.device, dtype=torch.double)

        self.unquantize_all()
        with torch.no_grad():
            for b in range(0, self.data.shape[0], self.batchsize):
                output[b:b+self.batchsize] = self.model(self.data[b:b+self.batchsize]).logits

        self.V = torch.linalg.eig(torch.einsum('bji,bjk->ik', output, output))[1].T[:self.length].real.half()
        
    def validate(self):
        losses = 0
        with torch.no_grad():
            for b in tqdm(range(0, self.data.shape[0], self.batchsize)):
                logits = self.model.lm_head(self.model(self.data[b:b+self.batchsize]).logits)[:,:-1].flatten(0,1)
                labels = self.labels[b:b+self.batchsize,1:].flatten(0,1)
                losses = losses + cross_entropy(logits.float(), labels, reduction='none').sum()

        ppl = torch.exp(losses / (self.data.shape[0] * (self.data.shape[1] - 1)))
        bit = self.model.bitrate()
        zer = self.model.numzero()
        print("perplexity: %f, bitrate: %f, numzero: %f" % (ppl, bit, zer))

        return ppl, bit

    def unquantize_all(self):
        self.model.unquantize_all()

    def quantize_all(self, log2_lambda=None, batches=None):
        batches = self.batches if batches is None else batches
        log2_lambda = self.loglambda if log2_lambda is None else log2_lambda

        self.model.quantize_all(log2_lambda + math.log2(batches))

    def load_vars(self):
        data = torch.load(self.load_file)
        for i in range(len(self.model.layers)):
            self.model.layers[i][1].grad_sq0 = data["%02d_grad_sq0" % i].to(self.device)
            self.model.layers[i][1].grad_sq1 = data["%02d_grad_sq1" % i].to(self.device)
            self.model.layers[i][1].weight_0 = data["%02d_weight_0" % i].to(self.device)
            self.model.layers[i][1].weight_1 = data["%02d_weight_1" % i].to(self.device)

    def save_vars(self):
        data = dict()
        for i in range(len(self.model.layers)):
            data["%02d_grad_sq0" % i] = self.model.layers[i][1].grad_sq0.detach_().cpu()
            data["%02d_grad_sq1" % i] = self.model.layers[i][1].grad_sq1.detach_().cpu()
            data["%02d_weight_0" % i] = self.model.layers[i][1].weight_0.detach_().cpu()
            data["%02d_weight_1" % i] = self.model.layers[i][1].weight_1.detach_().cpu()
        torch.save(data, self.save_file)

    @property
    def device(self):
        return self.model.device

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



        
