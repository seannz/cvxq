import math
import torch
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, Identity, Parameter, ReLU
from torch.nn import functional as F
from functools import reduce
from torch.distributions.laplace import Laplace
from transformers import BloomForCausalLM, OPTForCausalLM

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
        self.groups = groups
        self.name = name
        # self.bit_group2 = None

        self.register_buffer('input_av', torch.zeros([1, linear.weight.shape[1]], dtype=linear.bias.dtype))
        self.register_buffer('grad_sq0', torch.zeros([linear.weight.shape[0], 1], dtype=linear.bias.dtype))
        self.register_buffer('grad_sq2', torch.zeros([groups, linear.weight.shape[1]], dtype=linear.bias.dtype))
        # self.register_buffer('grad_sq2', torch.zeros([linear.weight.shape[0]//group, 1, linear.weight.shape[1]], dtype=torch.float))

        self.register_buffer('bit_group2', torch.arange(linear.weight.shape[0]).reshape([groups, -1]))

        # self.register_buffer('offset_0', torch.median(linear.weight.detach(), 1, keepdim=True)[0])
        # self.register_buffer('offset_1', linear.weight.detach().float().median(0, keepdim=True)[0])

        self.register_buffer('offset_2', torch.median(linear.weight.detach().unflatten(0, [groups, -1]), 1)[0])
        # self.register_buffer('offset_2', torch.median(linear.weight.detach(), 0, keepdim=True)[0])

        # self.register_buffer('weight_0', linear.weight.detach().float().var(1, keepdim=True))
        # self.register_buffer('weight_1', linear.weight.detach().float().var(0, keepdim=True))

        self.register_buffer('weight_q', linear.weight.detach())
        self.register_buffer('offset_q', torch.zeros([1, linear.weight.shape[1]], dtype=linear.bias.dtype))

    def forward(self, input: Tensor) -> Tensor:
        # if self.name == 'model.decoder.layers.10.self_attn.out_proj':
        #     print("forward", input.float().mean().item())

        return F.linear(input, self.weight_q, self.bias)

    def scale_sq(self) -> None:
        self.count_bw += 1

        self.grad_sq0 *= 1 - (1/self.count_bw)
        # self.grad_sq1 *= 1 - (1/self.count_bw)
        self.grad_sq2 *= 1 - (1/self.count_bw)

    def clearvar(self):
        # self.grad_sqr.zero_()
        self.grad_sq0.zero_()
        self.grad_sq2.zero_()
        self.input_av.zero_()
        self.count_fw = 0
        self.count_bw = 0

    def quantize(self):
        # bit_group = self.bit_group
        step_size = self.step_size.float()
        bit_depth = self.bit_depth.float()
        offset_2 = self.offset_2.float()
        # breakpoint()
        # if self.linear.weight.shape[0] > self.linear.weight.shape[1]:
        #     quants = self.linear.weight
        # else:
        # breakpoint()
        quants = self.linear.weight #[self.bit_group2] #.to(self.linear.weight.device)
        quants = Laplace(offset_2, (3/1.414213562) * 2. ** step_size).cdf(quants).multiply_(2. ** bit_depth)
        quants = quants.floor_().clamp_(None, 2. ** bit_depth - 1).add_(0.5)
        quants = Laplace(offset_2, (3/1.414213562) * 2. ** step_size).icdf(quants.multiply_(2. **-bit_depth))
        quants = quants.to(self.linear.weight.dtype)

        if quants.isnan().any():
            breakpoint()

        # if self.linear.weight.shape[0] > self.linear.weight.shape[1]:
        #     self.weight_q.copy_(quants)
        # else:
        # self.weight_q[bit_group] = quants
        self.weight_q.copy_(quants)

        # quants = self.linear.weight #.to(self.linear.weight.device)
        # quants = Laplace(self.offset_1, 0.98 * (3/1.414213562) * 2. ** step_size).cdf(quants).multiply(2. ** bit_depth)
        # quants = quants.floor().clamp(None, 2. ** bit_depth - 1).add(0.5)
        # quants = Laplace(self.offset_1, 0.98 * (3/1.414213562) * 2. ** step_size).icdf(quants.multiply(2. **-bit_depth))
        # quants = quants.to(self.linear.weight.dtype)

        # self.weight_q.copy_(quants)

    def unquantize(self):
        self.weight_q.copy_(self.linear.weight)
        # self.quantized = False

    @property
    def bit_group(self):
        bit_group = torch.var(self.linear.weight, 1, keepdim=True).log2().add_(self.grad_sq0.log2()) #.subtract(self.log2_lambda).multiply(0.5) # should be var?
        bit_group = torch.stack(bit_group.sort(0).indices.tensor_split(self.groups)).to(self.linear.weight.dtype) # stack so we can take means across each group

        self.bit_group2.copy_(bit_group) # torch.stack(self.bit_group2.sort(0).indices.tensor_split(self.group))) # stack so we can take means across each group

        return self.bit_group2

    @property
    def bit_depth(self):
        # self.bit_depth2 = torch.mean(self.linear.weight ** 2, 0, True).log2().add((self.grad_sq1).log2()).subtract(self.log2_lambda).multiply(0.5)
        # if self.linear.weight.shape[0] > self.linear.weight.shape[1]:
        #     self.bit_depth2 = torch.mean((self.linear.weight - self.offset_2) ** 2, 1, True).log2().add(torch.mean(self.grad_sqr, 1, True).log2()).subtract(self.log2_lambda).multiply(0.5)
        # else:
        # weight = self.linear.weight[self.bit_group2]
        # median = weight.median(1).values

        # self.offset_2.copy_(median) #self.linear.weight[self.bit_group2].median(1).values)
        self.bit_depth2 = torch.mean((self.linear.weight.float() - self.offset_2.float()) ** 2, 0, True).log2_().add_(self.grad_sq2.float().log2()).subtract_(self.log2_lambda).multiply_(0.5)
        # self.bit_depth2 = torch.mean((self.linear.weight[self.bit_group2] - self.offset_2) ** 2, 1, True).contiguous().log2().add(self.grad_sq2.log2()).subtract(self.log2_lambda).multiply(0.5)
        self.bit_depth2 = torch.threshold_(self.bit_depth2, 1.0, 0).round_().clamp_(0, 8).to(self.linear.weight.dtype)

        # self.bit_depth2 = self.newton(torch.mean((self.linear.weight - self.offset_1) ** 2, 0, True).log2().add((self.grad_sq1).log2())).round().clamp(0) # .log2().add((self.grad_sq1).log2()).subtract(self.log2_lambda).multiply(0.5).round().clamp(0)
        # self.bit_depth2 = torch.log2(2 ** self.bit_depth2.clamp(0) * (1 + 1) - 1).round()
        # self.bit_depth2 = self.bit_depth2.clamp(0).where(self.bit_depth2 == 0, self.bit_depth2.clamp(0) + 1).round()
        # self.bit_depth2[self.bit_depth2 < 3] = 0

        # self.bit_depth0 = ((self.weight_0.log2().add((self.grad_sq0).log2()).subtract(self.log2_lambda).multiply(0.5) + beta) ** 2 - beta ** 2).sqrt().round().clamp(0)





        # self.bit_depth1 = ((self.weight_1.log2().add((self.grad_sq1).log2()).subtract(self.log2_lambda).multiply(0.5) + beta) ** 2 - beta ** 2).sqrt().round().clamp(0)
        # self.bit_depth2 = torch.mean((self.linear.weight - self.offset_1) ** 2, 0, True).log2().add((self.grad_sq1).log2()).subtract(self.log2_lambda).multiply(0.5).round().clamp(0)


        # if self.bit_depth0.mean() < self.bit_depth1.mean():
        #     return self.bit_depth0

        # if self.weight_q.shape[0] > self.weight_q.shape[1]:
        #     return bit_depth0.round().clamp(0)

        # depths = depth0 + depth1
        # depthm = (2 ** (2 * depth0) + 2 ** (2 * depth1)).multiply(0.5).log2().multiply(0.5)
        # depth2 = depths.subtract_(depthm)

        return self.bit_depth2 #.round().clamp(0)  #, steps1] if depth1.mean() < depth0.mean() else [depth0, steps0]

    @property
    def step_size(self):
        # self.step_size0 = self.weight_0.log2().multiply(0.5)
        # self.step_size1 = self.weight_1.log2().multiply(0.5)

        # different step sizess for negative and positive weights
        # self.offset_2 = self.linear.weight[self.bit_group2].float().median(1, keepdim=True)[0]
        # if self.linear.weight.shape[0] > self.linear.weight.shape[1]:
        #     self.offset_2 = self.linear.weight.float().median(1, keepdim=True)[0]
        #     mask = self.linear.weight.gt(self.offset_2)
        #     step_sizep = torch.sum((self.linear.weight - self.offset_2) ** 2 * mask, 1, True) / torch.sum( mask, 1, True)
        #     step_sizem = torch.sum((self.linear.weight - self.offset_2) ** 2 *~mask, 1, True) / torch.sum(~mask, 1, True)
            
        # else:
        # self.offset_2 = self.linear.weight[self.bit_group2].float().median(1, keepdim=True)[0]
        # mask = self.linear.weight[self.bit_group2].gt(self.offset_2)
        # step_sizep = torch.sum((self.linear.weight[self.bit_group2] - self.offset_2) ** 2 * mask, 1, True) / torch.sum( mask, 1, True)
        # step_sizem = torch.sum((self.linear.weight[self.bit_group2] - self.offset_2) ** 2 *~mask, 1, True) / torch.sum(~mask, 1, True)

        # self.offset_2 = self.linear.weight.float().median(0, keepdim=True)[0]
        mask = self.linear.weight.gt(self.offset_2)
        weight_offset_2 = (self.linear.weight.float() - self.offset_2.float()) ** 2
        step_sizep = torch.sum(weight_offset_2 * mask, 0) / torch.sum( mask, 0)
        step_sizem = torch.sum(weight_offset_2 *~mask, 0) / torch.sum(~mask, 0)

        # breakpoint()
        # mask = self.linear.weight[self.bit_group2].gt(self.offset_2)
        # step_sizep = torch.sum((self.linear.weight[self.bit_group2] - self.offset_2) ** 2 * mask, 1, True) / torch.sum( mask, 1, True)
        # step_sizem = torch.sum((self.linear.weight[self.bit_group2] - self.offset_2) ** 2 *~mask, 1, True) / torch.sum(~mask, 1, True)

        # mask = self.linear.weight.gt(self.offset_1)
        # step_sizep = torch.sum((self.linear.weight - self.offset_1) ** 2 * mask, 0, True) / torch.sum( mask, 0, True)
        # step_sizem = torch.sum((self.linear.weight - self.offset_1) ** 2 *~mask, 0, True) / torch.sum(~mask, 0, True)

        self.step_size2 = torch.where(mask, step_sizep, step_sizem).log2().multiply(0.5).to(self.linear.weight.dtype)


        # if self.bit_depth0.mean() < self.bit_depth1.mean():
        #     return self.step_size0

        # if self.weight_q.shape[0] > self.weight_q.shape[1]:
        #     return step_size0

        # stepss = steps0 + steps1
        # stepsm = (2 ** (2 * steps0) + 2 ** (2 * steps1)).multiply(0.5).log2().multiply(0.5)
        # steps2 = stepss.subtract(stepsm)

        return self.step_size2

    @property
    def bias(self):
        # return self.linear.bias - ((self.weight_q - self.linear.weight).float() @ self.offset_q.squeeze()).to(self.linear.bias.dtype)
        return self.linear.bias - ((self.weight_q.float() - self.linear.weight.float()) @ self.offset_q.squeeze().float()).to(self.linear.bias.dtype) #.squeeze())

    def forward_hook_dbg(self, module, layer_in, layer_out) -> None:
        None # print(layer_in[0].float().mean().item())
        # breakpoint()
        # print("forward " + str(self.name))

    def forward_hook(self, module, layer_in, layer_out) -> None:
        self.layer_in = layer_in[0].detach().squeeze()
        input_av = self.layer_in # torch.flatten(self.layer_in, 0, 1)

        self.count_fw += 1
        self.input_av += (1/self.count_fw) * (input_av.float().mean(0, True) - self.input_av.float()) #).to(self.input_av.dtype)
        # print("input av", str(self.name), self.input_av.float().mean().item())

    def backward_hook_dbg(self, module, grad_in, grad_out) -> None:
        None # print(grad_out[0].float().mean().item())
        # print("backward " + str(self.name))

    def backward_hook(self, module, grad_in, grad_out) -> None:
        self.grad_out = grad_out[0].detach().squeeze()
        grad_sqr = torch.einsum("ij,ik->jk", self.grad_out, self.layer_in).float() # works better on -125m with wikitext-2, 

        self.grad_sq0 += (1/self.count_bw) * grad_sqr.square().mean(1, True) #.contiguous()
        # self.grad_sq1 += (1/self.count_bw) * grad_sqr.square().mean(0, True)
        # self.grad_sq2 += (1/self.count_bw) * grad_sqr.square()[self.bit_group2].mean(1, True).contiguous() # Sean change
        self.grad_sq2 += (1/self.count_bw) * grad_sqr.square().mean(0, True) #.contiguous() # Sean change

        self.grad_out = None

    def add_forward_hooks(self) -> None:
        # self.hooks = [self.linear.weight.register_hook(self.weight_hook)]
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
        self.offset_q.copy_(self.input_av) #.clone()

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

class ModuleQ(Module):
    def __init__(self, model, layer_ind=0, num_layers=-1, groups=1, checkpointing=False) -> None:
        super().__init__()
        self.lm_head = model.lm_head #.requires_grad_(False)
        
        if isinstance(model, OPTForCausalLM):
            self.embed_tokens = model.model.decoder.embed_tokens #.requires_grad_(False)
            self.n_head = model.config.num_attention_heads
            self.groups = groups # model.config.hidden_size // groups

        if isinstance(model, BloomForCausalLM):
            self.embed_tokens = model.transformer.word_embeddings
            self.n_head = model.config.n_head
            self.embed_dimens = None

        self.model = model.eval().requires_grad_(False)

        model.lm_head = Identity()
        # model.model.decoder.embed_tokens = Identity()

        if checkpointing:
            model.gradient_checkpointing_enable()

        # layers = [(n, m) for n, m in model.named_modules() if n.endswith('query_key_value')]
        # for layer in layers:
        #     setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], Linear3(layer[1], self.n_head))

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], LinearQ(layer[1], self.groups))
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

    def numzero(self):
        zeros = numel = 0
        for i in range(len(self.layers)):
            # breakpoint()
            zeros += self.layers[i][1].bit_depth.count_nonzero()
            numel += self.layers[i][1].bit_depth.numel()

        return (1 - zeros / numel).item()

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
            bits += self.layers[i][1].bit_depth.float().mean().item() * self.layers[i][1].linear.weight.numel()
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

    def quantize_all(self, log2_lambda) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].log2_lambda = log2_lambda
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
        # for i in range(len(self.actfns)):
        #     self.actfns[i][1].add_backward_hooks()

    def remove_backward_hooks(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].remove_backward_hooks()

    def update_offsets(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].update_offsets()

    def add_hooks_dbg(self) -> None:
        # for i in range(len(self.layers)):
        self.layers[66][1].add_hooks_dbg()

    def remove_hooks_dbg(self) -> None:
        # for i in range(len(self.layers)):
        self.layers[66][1].remove_hooks_dbg()

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config
