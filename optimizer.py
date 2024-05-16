import torch
from torch.nn import Module, Linear, Identity
from torch.nn.functional import mse_loss, cross_entropy

from tqdm import tqdm
from scipy.io import savemat, loadmat

class Optimizer(Module):
    def __init__(self, model, data, labels, batches, batchsize=1, stride=1, loglambda=-18, model_id="unknown", load_file=None, save_file=None) -> None:
        super().__init__()
        self.model = model
        self.model_id = "_".join(model_id.split("/"))
        self.loglambda = loglambda
        self.data = data
        self.labels = labels
        self.batches = batches
        self.batchsize = batchsize
        self.stride = stride
        self.load_file = load_file
        self.save_file = save_file

        if self.load_file is None: #
            return

        for i in range(len(self.model.layers)):
            gradsq = torch.as_tensor(loadmat(self.load_file % i)['gradsq'], device=self.model.device)
            self.model.layers[i][1].gradsq = gradsq

    def calibrate(self):
        # backward call to populate the grad squared
        self.model.add_hooks()
        for b in range(0, self.batches, self.batchsize):
            self.model.unquantize_all()
            for t in tqdm(list(reversed(range(0,self.data.shape[-2],self.stride)))):
                output = self.model(self.data[b:b+self.batchsize,:t+1]) #.logits
                for s in range(output.logits.shape[-1]):
                    # self.model.zero_grad()
                    output.logits[:,t,s].sum().backward(retain_graph = True)
                del output

            self.optimize(self.loglambda, b + self.batchsize)
            self.validate()

        self.model.remove_hooks()

        if self.save_file is None:
            return

        for i in range(len(self.model.layers)):
            savemat(self.save_file % i, {"gradsq": self.model.layers[i][1].gradsq.detach_().cpu().numpy(),
                                         "weight": self.model.layers[i][1].linear.weight.detach_().cpu().numpy()})

    def validate(self):
        losses = 0
        with torch.no_grad():
            for b in range(0, self.data.shape[0], self.batchsize):
                logits = self.model.lm_head(self.model(self.data[b:b+self.batchsize]).logits)[:,:-1].flatten(0,1)
                labels = self.labels[b:b+self.batchsize,1:].flatten(0,1)
                losses = losses + cross_entropy(logits.float(), labels, reduction='none').sum()

        ppl = torch.exp(losses / (self.data.shape[0] * (self.data.shape[1] - 1)))
        bit = self.model.bitrate()
        print("perplexity: %f, bitrate: %f" % (ppl, bit))

        return ppl, bit

    def optimize(self, log2lam, numsq=None):
        numsq = self.batches if numsq is None else numsq
        # optimize bit depths for a given lambda
        with torch.no_grad():
            for i in range(len(self.model.layers)):
                weight = torch.stack([self.model.layers[i][1].gradsq * (1/numsq), self.model.layers[i][1].linear.weight.float().square()])

                depth0 = weight.mean(1, keepdims=True).log2_().sum(0).subtract_(log2lam).multiply_(0.5)
                depth1 = weight.mean(2, keepdims=True).log2_().sum(0).subtract_(log2lam).multiply_(0.5)
                depths = depth0 + depth1
                depthm = (2 ** (2 * depth0) + 2 ** (2 * depth1)).multiply_(0.5).log2_().multiply_(0.5)
                depth2 = depths.subtract_(depthm)

                steps0 = weight[1].mean(0, keepdims=True).log2_().multiply_(0.5)
                steps1 = weight[1].mean(1, keepdims=True).log2_().multiply_(0.5)
                stepss = steps0 + steps1
                stepsm = (2 ** (2 * steps0) + 2 ** (2 * steps1)).multiply_(0.5).log2_().multiply_(0.5)
                steps2 = stepss.subtract_(stepsm)

                # self.model.layers[i][1].bit_depth[:,:] = depth2.floor().add_(depth2.frac_().gt_(0.44313)).clamp_(0)
                self.model.layers[i][1].bit_depth[:,:] = depth2.round_().clamp_(0)
                self.model.layers[i][1].step_size[:,:] = steps2 # depths.subtract_(depthm).round_().clamp_(0)

        self.model.quantize_all()

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



        
