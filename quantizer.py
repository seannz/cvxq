import torch
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, Identity
from torch.nn import functional as F
from functools import reduce
from torch.distributions.laplace import Laplace

class Quantizer(Module):
    def __init__(self, quantized=False, log2_lambda=-18) -> None:
        super().__init__()
        self.quantized = quantized
        self.log2_lambda = log2_lambda

    def forward(self, params) -> Tensor:
        raise NotImplementedError

        # if not self.quantized:
        #     return params

        # quants = Laplace(0, (3/1.4142) * 2. ** self.step_size).cdf(params).multiply_(2. ** self.bit_depth)
        # quants = quants.floor_().clamp_(None, 2. ** self.bit_depth - 1).add_(0.5)
        # quants = Laplace(0, (3/1.4142) * 2. ** self.step_size).icdf(quants.multiply_(2. **-self.bit_depth))
        # quants = params

        # return quants.to(params.dtype) # quants # params # quants

    def quantize(self):
        raise NotImplementedError

    @property
    def bit_depth(self):
        raise NotImplementedError

    @property
    def step_size(self):
        raise NotImplementedError

class LinearQ(Quantizer):
    def __init__(self, linear, momentum=0.1) -> None:
        super().__init__()
        self.linear = linear
        self.momentum = momentum

        self.register_buffer('grad_sq0', torch.zeros([linear.weight.shape[0], 1], dtype=torch.double))
        self.register_buffer('grad_sq1', torch.zeros([1, linear.weight.shape[1]], dtype=torch.double))
        self.register_buffer('weight_0', linear.weight.detach().double().square().mean(1, keepdim=True))
        self.register_buffer('weight_1', linear.weight.detach().double().square().mean(0, keepdim=True))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias) # - F.linear(self.inputs.half(), self.weight - self.linear.weight)

    def quantize(self):
        quants = self.linear.weight

        quants = Laplace(0, (3/1.4142) * 2. ** self.step_size).cdf(quants).multiply_(2. ** self.bit_depth)
        quants = quants.floor_().clamp_(None, 2. ** self.bit_depth - 1).add_(0.5)
        quants = Laplace(0, (3/1.4142) * 2. ** self.step_size).icdf(quants.multiply_(2. **-self.bit_depth))
        quants = quants.to(self.linear.weight.dtype)

        self.weightq = quants
        self.quantized = True

    def unquantize(self):
        self.weightq = None
        self.quantized = False

    @property
    def bit_depth(self):
        depth0 = self.weight_0.log2().add_(self.grad_sq0.log2()).subtract_(self.log2_lambda).multiply_(0.5)
        depth1 = self.weight_1.log2().add_(self.grad_sq1.log2()).subtract_(self.log2_lambda).multiply_(0.5)
        depths = depth0 + depth1
        depthm = (2 ** (2 * depth0) + 2 ** (2 * depth1)).multiply_(0.5).log2_().multiply_(0.5)
        depth2 = depths.subtract_(depthm)
        # return depth2.floor().add_(depth2.frac_().gt_(0.44313)).clamp_(0)
        return depth1.round_().clamp_(0)

    @property
    def step_size(self):
        steps0 = self.weight_0.log2().multiply_(0.5)
        steps1 = self.weight_1.log2().multiply_(0.5)
        stepss = steps0 + steps1
        stepsm = (2 ** (2 * steps0) + 2 ** (2 * steps1)).multiply_(0.5).log2_().multiply_(0.5)
        steps2 = stepss.subtract_(stepsm)

        return steps1

    @property
    def weight(self):
        return self.weightq if self.quantized else self.linear.weight

    @property
    def bias(self):
        return self.linear.bias # - (self.weight - self.linear.weight) @ self.inputs.half().squeeze()

    def forward_hook(self, module, layer_in, layer_out) -> None:
        self.layer_in = layer_in[0].detach()

    def backward_hook(self, module, grad_in, grad_out) -> None:
        self.grad_out = grad_out[0].detach()

        grad_sqr = torch.einsum("bij,bik->bjk", self.grad_out, self.layer_in).double().square_().sum(0)

        self.grad_sq0 += grad_sqr.double().mean(1, keepdims=True)
        self.grad_sq1 += grad_sqr.double().mean(0, keepdims=True)

        self.grad_out = None

        # print(self.gradsq.sum())
        # breakpoint()
        # if self.grad_out[:,1025:].square().sum().item() != 0:
        #     print('detected non-zero grad beyond the current token')

    # def weight_hook(self, grad) -> None:
    #     grad_sqr = grad.float().square()

    def add_hooks(self) -> None:
        # self.hooks = [self.linear.weight.register_hook(self.weight_hook)]
        self.hooks = [self.register_forward_hook(self.forward_hook), self.register_full_backward_hook(self.backward_hook)]

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __repr__(self) -> str:
        return "LinearQ(%s)" % self.linear.extra_repr()

class ModuleQ(Module):
    def __init__(self, model, checkpointing=False) -> None:
        super().__init__()
        self.lm_head = model.lm_head #.requires_grad_(False)
        self.embed_tokens = model.model.decoder.embed_tokens #.requires_grad_(False)
        self.model = model.eval().requires_grad_(False)
        # self.embed_positions = model.model.decoder.embed_positions #.requires_grad_(False)

        model.lm_head = Identity()
        model.model.decoder.embed_tokens = Identity()

        if checkpointing:
            model.gradient_checkpointing_enable()

        # layers = [(n.split(sep='.'), m) for n, m in model.named_modules() if isinstance(m, LayerNorm)]
        # for layer in layers:
        #     layer[1].requires_grad_(False)

        layers = [(n.split(sep='.'), m) for n, m in model.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            # layer[1].requires_grad_(True)
            setattr(reduce(getattr, layer[0][:-1], model), layer[0][-1], LinearQ(layer[1], layer[0]))

        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

    def numzero(self):
        zeros = numel = 0
        for i in range(len(self.layers)):
            # breakpoint()
            zeros += self.layers[i][1].bit_depth.count_nonzero()
            numel += self.layers[i][1].bit_depth.numel()

        return 1 - zeros / numel

    def bitrate(self):
        # breakpoint()
        bits = numel = 0
        for i in range(len(self.layers)):
            # breakpoint()
            bits += self.layers[i][1].bit_depth.mean().float() * self.layers[i][1].linear.weight.numel()
            numel += self.layers[i][1].linear.weight.numel()

        return bits / numel

    def forward(self, input: Tensor, attention_mask: Tensor = None) -> Tensor:
        return self.model.forward(inputs_embeds=input)

    def quantize_all(self, log2_lambda) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].log2_lambda = log2_lambda
            self.layers[i][1].quantize() # = True

    def unquantize_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].unquantize() # = False

    def add_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].add_hooks()

    def remove_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].remove_hooks()

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config
