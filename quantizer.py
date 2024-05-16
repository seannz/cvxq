import torch
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, Identity
from torch.nn import functional as F
from functools import reduce
from torch.distributions.laplace import Laplace

class Quantizer(Module):
    def __init__(self, params, quantized=False) -> None:
        super().__init__()
        self.quantized = quantized
        self.register_buffer('bit_depth', torch.inf * torch.ones_like(params))
        self.register_buffer('step_size', torch.zeros_like(params))

    def forward(self, params) -> Tensor:
        if not self.quantized:
            return params

        quants = Laplace(0, (3/1.4142) * 2. ** self.step_size).cdf(params).multiply_(2. ** self.bit_depth)
        quants = quants.floor_().clamp_(None, 2. ** self.bit_depth - 1).add_(0.5)
        quants = Laplace(0, (3/1.4142) * 2. ** self.step_size).icdf(quants.multiply_(2. **-self.bit_depth))

        return quants # quants # params # quants

    def extra_repr(self) -> str:
        return f'bit_depth={self.bit_depth}, step_size={self.step_size}'


class LinearQ(Quantizer):
    def __init__(self, linear, name=None) -> None:
        super().__init__(linear.weight)
        self.linear = linear
        # self.linear.weight.requires_grad_(True)

        self.register_buffer('gradsq', torch.zeros(linear.weight.shape, dtype=torch.float))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias) # - F.linear(self.inputs.half(), self.weight - self.linear.weight)

    @property
    def weight(self):
        return super().forward(self.linear.weight) #.detach()

    @property
    def bias(self):
        return self.linear.bias # - (self.weight - self.linear.weight) @ self.inputs.half().squeeze()

    def forward_hook(self, module, layer_in, layer_out) -> None:
        self.layer_in = layer_in[0].detach()

    def backward_hook(self, module, grad_in, grad_out) -> None:
        self.grad_out = grad_out[0].detach()
        self.gradsq += torch.einsum("bij,bik->bjk", self.grad_out, self.layer_in).float().square_().sum(0)
        # print(self.gradsq.sum())
        # breakpoint()
        # if self.grad_out[:,1025:].square().sum().item() != 0:
        #     print('detected non-zero grad beyond the current token')

    def weight_hook(self, grad) -> None:
        self.gradsq += grad.float().square()

    def add_hooks(self) -> None:
        # self.hooks = [self.linear.weight.register_hook(self.weight_hook)]
        self.hooks = [self.register_forward_hook(self.forward_hook), self.register_full_backward_hook(self.backward_hook)]

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __repr__(self) -> str:
        return "LinearQ(%s)" % self.linear.extra_repr()

class ModuleQ(Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.lm_head = model.lm_head #.requires_grad_(False)
        self.embed_tokens = model.model.decoder.embed_tokens #.requires_grad_(False)
        self.model = model.eval().requires_grad_(False)
        # self.embed_positions = model.model.decoder.embed_positions #.requires_grad_(False)

        model.lm_head = Identity()
        model.model.decoder.embed_tokens = Identity()

        # layers = [(n.split(sep='.'), m) for n, m in model.named_modules() if isinstance(m, LayerNorm)]
        # for layer in layers:
        #     layer[1].requires_grad_(False)

        layers = [(n.split(sep='.'), m) for n, m in model.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            # layer[1].requires_grad_(True)
            setattr(reduce(getattr, layer[0][:-1], model), layer[0][-1], LinearQ(layer[1], layer[0]))

        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

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

    def quantize_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].quantized = True

    def unquantize_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].quantized = False

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
