import copy
import math
import torch
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, Identity, Parameter, ReLU
from torch.nn import functional as F
from functools import reduce
from torch.distributions.laplace import Laplace
from transformers import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTDecoderLayer

class Quantizer(Module):
    def __init__(self, quantized=False, log2_lambda=-18) -> None:
        super().__init__()
        self.quantized = quantized
        self.log2_lambda = log2_lambda

    def forward(self, params) -> Tensor:
        raise NotImplementedError

    def quantize(self):
        raise NotImplementedError

    @property
    def bit_depth(self):
        raise NotImplementedError

    @property
    def step_size(self):
        raise NotImplementedError

class LinearQ(Quantizer):
    def __init__(self, linear, groups=1, name=None) -> None:
        super().__init__()
        self.linear = linear
        self.count_fw = 0
        self.count_bw = 0
        # self.count_qt = 0
        self.groups = groups
        self.name = name
        # self.dropin = dropin

        self.register_buffer('input_av', torch.zeros([1, linear.weight.shape[1]], dtype=linear.bias.dtype))

        self.register_buffer('grad_sq0', torch.zeros([linear.weight.shape[0], 1], dtype=linear.bias.dtype))
        # self.register_buffer('grad_sq1', torch.zeros([1, linear.weight.shape[1]], dtype=linear.bias.dtype))
        self.register_buffer('grad_sq2', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.bias.dtype))
        self.register_buffer('groups_2', torch.arange(linear.weight.shape[0]).reshape([groups, -1]))

        # self.register_buffer('grad_sq3', torch.zeros([linear.weight.shape[0], groups, 1], dtype=linear.bias.dtype))
        # self.register_buffer('groups_3', torch.arange(linear.weight.shape[1]).reshape([groups, -1]))

        self.register_buffer('weight', linear.weight.detach())
        self.register_buffer('offset', linear.bias.detach())
        
        # self.register_buffer('step_sizep', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.bias.dtype))
        # self.register_buffer('step_sizem', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.bias.dtype))

        # self.register_buffer('bit_depth1', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.bias.dtype))
        self.register_buffer('bit_depth2', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.bias.dtype))
        # self.register_buffer('bit_depth3', torch.zeros([linear.weight.shape[0], groups, 1], dtype=linear.bias.dtype))

    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.offset)

        return output

    def scale_sq(self) -> None:
        self.count_bw += 1
        self.grad_sq0 *= 1 - (1/(self.count_bw))
        self.grad_sq2 *= 1 - (1/(self.count_bw))

    # def clearvar(self):
    #     self.grad_sqr.zero_()
    #     self.grad_sq0.zero_()
    #     self.grad_sq1.zero_()
    #     self.grad_sq2.zero_()
    #     self.input_av.zero_()
    #     self.count_fw = 16
    #     self.count_bw = 16

    def optimize(self):
        groups = self.bit_group()
        weight = self.linear.weight[groups] #.float()
        offset = torch.median(weight, 1, keepdim=True).values

        bit_depth = self.bit_depth((weight - offset).float()) #.float()

        self.groups_2.copy_(groups.to(self.groups_2.dtype))
        # self.bit_depth2.copy_(bit_depth.to(self.bit_depth2.dtype))
        self.bit_depth2.copy_(bit_depth.to(self.bit_depth2.dtype))
        # self.bit_depth2.add_(bit_depth.to(self.bit_depth2.dtype)).multiply_(0.5)

    def quantize(self):
        groups = self.groups_2 #.bit_group()
        weight = self.linear.weight[groups]
        offset = torch.median(weight, 1, keepdim=True).values

        # self.count_qt += 1
        # bit_depth = self.bit_depth1.add_((1/self.count_qt) * (self.bit_depth2 - self.bit_depth1)).round().float()
        bit_depth = self.bit_depth2.round().float() #1.add_((1/self.count_qt) * (self.bit_depth2 - self.bit_depth1)).round().float()

        step_size = self.step_size((weight - offset).float())

        quants = Laplace(offset, (3/1.414213562) * 2. ** step_size).cdf(weight).multiply_(2. ** bit_depth)
        quants = quants.floor_().clamp_(None, 2. ** bit_depth - 1).add_(0.5)
        quants = Laplace(offset, (3/1.414213562) * 2. ** step_size).icdf(quants.multiply_(2. **-bit_depth))

        self.weight[groups] = quants.to(self.weight.dtype)

        # introduce error drop-out
        # if dropout > 0:
        #     self.weight.copy_(self.weight.where(1 - torch.rand(self.weight.shape, device=quants.device) > dropout, self.linear.weight))

        if quants.isnan().any():
            breakpoint()
        
        # self.groups_2.copy_(groups.to(self.groups_2.dtype))
        # self.bit_depth2.copy_(bit_depth.to(self.bit_depth2.dtype))

    def unquantize(self):
        self.weight.copy_(self.linear.weight)

    def bit_group(self):
        bit_group = torch.var(self.linear.weight.float(), 1, keepdim=True).log2_().add_(self.grad_sq0.float().log2())
        bit_group = torch.stack(bit_group.squeeze(1).sort(0).indices.tensor_split(self.groups)).sort(1).values

        return bit_group

    def bit_depth(self, weight):
        bit_depth = torch.mean(weight ** 2, 1, keepdim=True).float().log2_().add_(self.grad_sq2.float().log2()).subtract_(self.log2_lambda).multiply_(0.5)
        # bit_depth = torch.mean(weight ** 2, 1, keepdim=True, dtype=torch.float).log2_().add_(self.grad_sq2.float().log2()).subtract_(self.log2_lambda).multiply_(0.5)
        bit_depth = torch.clamp_(bit_depth, 0, 8) #.round_()
        # bit_depth = torch.threshold_(bit_depth, 0.0, 0).round_() #.clamp_(0, 8) #.to(weight.dtype)

        return bit_depth

    def step_size(self, weight):
        step_sizp = torch.sum(weight ** 2 * weight.gt(0), 1, keepdim=True, dtype=torch.float).divide_(torch.sum( weight.gt(0), 1, keepdim=True, dtype=torch.float))
        step_sizm = torch.sum(weight ** 2 *~weight.gt(0), 1, keepdim=True, dtype=torch.float).divide_(torch.sum(~weight.gt(0), 1, keepdim=True, dtype=torch.float))
        step_size = torch.where(weight.gt(0), step_sizp, step_sizm).log2_().multiply_(0.5)

        return step_size #p, step_sizm

    def forward_hook(self, module, layer_in, layer_out) -> None:
        self.layer_in = layer_in[0].detach().squeeze()
        input_av = self.layer_in # torch.flatten(self.layer_in, 0, 1)

        self.count_fw += 1
        self.input_av += (1/(self.count_fw)) * (input_av.mean(0, True, dtype=torch.float).subtract_(self.input_av))

    def backward_hook(self, module, grad_in, grad_out) -> None:
        self.grad_out = grad_out[0].detach().squeeze()
        grad_sqr = torch.einsum("ij,ik->jk", self.grad_out, self.layer_in).float().square()

        self.grad_sq0 += (1/(self.count_bw)) * grad_sqr.mean(1, True, dtype=torch.float)
        # self.grad_sq1 += (1/self.count_bw) * grad_sqr.mean(0, True, dtype=torch.float)
        self.grad_sq2 += (1/(self.count_bw)) * grad_sqr[self.groups_2].mean(1, True, dtype=torch.float)

        if self.grad_out.isnan().any():
            breakpoint()

        if grad_in[0].isnan().any():
            breakpoint()

        self.grad_out = None

    def add_forward_hooks(self) -> None:
        self.forward_hooks = [self.register_forward_hook(self.forward_hook)]

    def add_hooks_dbg(self) -> None:
        self.forward_hooks_dbg = [self.register_forward_hook(self.forward_hook_dbg)]
        self.backward_hooks_dbg = [self.register_full_backward_hook(self.backward_hook_dbg)]

    def add_backward_hooks(self) -> None:
        self.backward_hooks = [self.register_full_backward_hook(self.backward_hook)]

    def remove_forward_hooks(self) -> None:
        for hook in self.forward_hooks:
            hook.remove()

    def update_offsets(self) -> None:
        self.offset.copy_(self.linear.bias - ((self.weight - self.linear.weight) @ self.input_av.squeeze()).to(self.linear.bias.dtype)) #.squeeze())

    def remove_backward_hooks(self) -> None:
        for hook in self.backward_hooks:
            hook.remove()

    def remove_hooks_dbg(self) -> None:
        for hook in self.forward_hooks_dbg:
            hook.remove()

        for hook in self.backward_hooks_dbg:
            hook.remove()

    def __repr__(self) -> str:
        return "LinearQ(%s)" % self.linear.extra_repr()

class BlockQ(Module):
    def __init__(self, block, name=None) -> None:
        super().__init__()
        self.block = block

    def forward(self, input: Tensor, attention_mask: Tensor, *args, **kwargs) -> Tensor:
        device = self.block.fc1.weight.device
        output = self.block(input.to(device), attention_mask.to(device), *args, **kwargs)

        return output

class ModuleQ(Module):
    def __init__(self, model, layer_ind=0, num_layers=-1, groups=1, checkpointing=False, gpus=1) -> None:
        super().__init__()
        if isinstance(model, OPTForCausalLM):
            self.embed_tokens = model.model.decoder.embed_tokens #.requires_grad_(False)
            self.n_head = model.config.num_attention_heads
            self.embed_dimens = model.config.hidden_size # // groups
            self.groups = groups

        if isinstance(model, BloomForCausalLM):
            self.embed_tokens = model.transformer.word_embeddings
            self.n_head = model.config.n_head
            self.embed_dimens = None
            self.groups = groups

        self.model = model.eval().requires_grad_(False)
        self.lm_head = copy.deepcopy(model.lm_head).cuda(gpus - 1) #.requires_grad_(False)
        model.lm_head = Identity()

        if checkpointing:
            model.gradient_checkpointing_enable()

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            groups = self.groups * (layer[1].out_features // self.embed_dimens)
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], LinearQ(layer[1], groups))
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, OPTDecoderLayer)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], BlockQ(layer[1]))
        self.blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, BlockQ)]

        model.model.decoder.embed_tokens.cuda(0)
        model.model.decoder.embed_positions.cuda(0)
        for i in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[i].cuda(int(i / len(model.model.decoder.layers) * gpus))
        model.model.decoder.final_layer_norm.cuda(gpus - 1)
        # breakpoint()
    def numzero(self):
        zeros = numel = 0
        for i in range(len(self.layers)):
            # breakpoint()
            zeros += self.layers[i][1].bit_depth2.count_nonzero().item()
            numel += self.layers[i][1].bit_depth2.numel()

        return 1 - zeros / numel

    def log2lam(self):
        lambs = 0
        for i in range(len(self.layers)):
            lambs += self.layers[i][1].log2_lambda

        return lambs / len(self.layers)

    def bitrate(self):
        # breakpoint()
        bits = numel = 0
        for i in range(len(self.layers)):
            # breakpoint()
            bits += self.layers[i][1].bit_depth2.round().mean(dtype=torch.float).item() * self.layers[i][1].linear.weight.numel()
            numel += self.layers[i][1].linear.weight.numel()

        return bits / numel

    def scale_sq(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].scale_sq()

    def forward(self, input: Tensor) -> Tensor:
        return self.model.forward(inputs_embeds=input)

    def bitgroup_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].bit_group #clearvar() # = True

    def clearvar_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].clearvar() # = True

    def optimize_all(self, log2_lambda) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].log2_lambda = log2_lambda
            self.layers[i][1].optimize() # = True

    def quantize_all(self, log2_lambda, skip=0, stride=1) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].quantize() # = True

    def unquantize_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].unquantize() # = False

    def add_forward_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].add_forward_hooks()

    def remove_forward_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].remove_forward_hooks()

    def add_backward_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].add_backward_hooks()

    def remove_backward_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].remove_backward_hooks()

    def update_offsets(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].update_offsets()

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config
