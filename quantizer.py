import torch
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, Identity, Parameter
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

class Linear3(Module):
    def __init__(self, linear, n_head) -> None:
        super().__init__()
        self.n_head = n_head
        self.hidden_size = linear.in_features

        weight = linear.weight.reshape(n_head,3,-1,linear.in_features)
        bias = linear.bias.reshape(n_head,3,-1)

        self.q_proj = Linear(self.hidden_size, self.hidden_size, device=weight.device).to(weight.dtype)
        with torch.no_grad():
            self.q_proj.weight.copy_(weight[:,0].reshape(-1,self.hidden_size)).requires_grad_(weight.requires_grad)
            self.q_proj.bias.copy_(bias[:,0].reshape(-1)).requires_grad_(bias.requires_grad)

        self.k_proj = Linear(self.hidden_size, self.hidden_size, device=weight.device).to(weight.dtype)
        with torch.no_grad():
            self.k_proj.weight.copy_(weight[:,1].reshape(-1,self.hidden_size)).requires_grad_(weight.requires_grad)
            self.k_proj.bias.copy_(bias[:,1].reshape(-1)).requires_grad_(bias.requires_grad)

        self.v_proj = Linear(self.hidden_size, self.hidden_size, device=weight.device).to(weight.dtype)
        with torch.no_grad():
            self.v_proj.weight.copy_(weight[:,2].reshape(-1,self.hidden_size)).requires_grad_(weight.requires_grad)
            self.v_proj.bias.copy_(bias[:,2].reshape(-1)).requires_grad_(bias.requires_grad)

    def forward(self, input) -> None:
        batch_size, seq_length = input.shape[0:2]

        q_proj = self.q_proj(input).reshape(batch_size, seq_length, self.n_head, -1)
        k_proj = self.k_proj(input).reshape(batch_size, seq_length, self.n_head, -1)
        v_proj = self.v_proj(input).reshape(batch_size, seq_length, self.n_head, -1)

        fused_qkv = torch.stack([q_proj, k_proj, v_proj], 3).flatten(2,4)

        return fused_qkv

class LinearQ(Quantizer):
    def __init__(self, linear, name=None) -> None:
        super().__init__()
        self.linear = linear
        self.count_fw = 0
        self.count_bw = 0
        self.name = name

        self.register_buffer('input_av', torch.zeros([1, linear.weight.shape[1]], dtype=torch.float))
        self.register_buffer('grad_sq0', torch.zeros([linear.weight.shape[0], 1], dtype=torch.float))
        self.register_buffer('grad_sq1', torch.zeros([1, linear.weight.shape[1]], dtype=torch.float))

        self.register_buffer('grad_av0', torch.zeros([linear.weight.shape[0], 1], dtype=torch.float))
        self.register_buffer('grad_av1', torch.zeros([1, linear.weight.shape[1]], dtype=torch.float))

        self.register_buffer('weight_0', linear.weight.detach().float().var(1, keepdim=True))
        self.register_buffer('weight_1', linear.weight.detach().float().var(0, keepdim=True))

        self.register_buffer('offset_0', linear.weight.detach().float().mean(1, keepdim=True))
        self.register_buffer('offset_1', linear.weight.detach().float().mean(0, keepdim=True))

        self.register_buffer('weight_q', linear.weight.detach())
        self.register_buffer('offset_q', torch.zeros([1, linear.weight.shape[1]], dtype=torch.float))

    def forward(self, input: Tensor) -> Tensor:
        # if self.name == 'model.decoder.layers.10.self_attn.out_proj':
        #     print("forward", input.float().mean().item())

        return F.linear(input, self.weight_q, self.bias)

    def scale_sq(self) -> None:
        self.count_bw += 1
        self.grad_av0 *= 1 - (1/self.count_bw)
        self.grad_av1 *= 1 - (1/self.count_bw)

        self.grad_sq0 *= 1 - (1/self.count_bw)
        self.grad_sq1 *= 1 - (1/self.count_bw)

    def quantize(self):
        bit_depth = self.bit_depth
        step_size = self.step_size

        quants = self.linear.weight #.to(self.linear.weight.device)
        quants = Laplace(self.offset_1, 0.98 * (3/1.414213562) * 2. ** step_size).cdf(quants).multiply(2. ** bit_depth)
        quants = quants.floor().clamp(None, 2. ** bit_depth - 1).add(0.5)
        quants = Laplace(self.offset_1, 0.98 * (3/1.414213562) * 2. ** step_size).icdf(quants.multiply(2. **-bit_depth))
        quants = quants.to(self.linear.weight.dtype)

        self.weight_q.copy_(quants)
        # self.quantized = True

        # if self.name == 'model.decoder.layers.11.fc2':  #        if self.name == 'model.decoder.layers.10.self_attn.out_proj':
        #     print("quantize", self.weight.float().mean().item(), bit_depth.float().mean().item())

    def unquantize(self):
        self.weight_q.copy_(self.linear.weight)
        # self.quantized = False


    # @property
    # def dim_quant(self):
    #     depth0 = self.weight_0.log2().add(self.grad_sq0.log2()).subtract(self.log2_lambda).multiply(0.5)
    #     depth1 = self.weight_1.log2().add(self.grad_sq1.log2()).subtract(self.log2_lambda).multiply(0.5)

    #     return 0 if depth0.clamp(0).mean() < depth1.clamp(0).mean() else 1

    @property
    def bit_depth(self):
        self.bit_depth0 = self.weight_0.log2().add((self.grad_sq0).log2()).subtract(self.log2_lambda).multiply(0.5).round().clamp(0)
        self.bit_depth1 = self.weight_1.log2().add((self.grad_sq1).log2()).subtract(self.log2_lambda).multiply(0.5).round().clamp(0)

        # if self.bit_depth0.mean() < self.bit_depth1.mean():
        #     return self.bit_depth0

        # if self.weight_q.shape[0] > self.weight_q.shape[1]:
        #     return bit_depth0.round().clamp(0)

        # depths = depth0 + depth1
        # depthm = (2 ** (2 * depth0) + 2 ** (2 * depth1)).multiply(0.5).log2().multiply(0.5)
        # depth2 = depths.subtract_(depthm)

        return self.bit_depth1 #.round().clamp(0)  #, steps1] if depth1.mean() < depth0.mean() else [depth0, steps0]

    @property
    def step_size(self):
        self.step_size0 = self.weight_0.log2().multiply(0.5)
        self.step_size1 = self.weight_1.log2().multiply(0.5)

        # if self.bit_depth0.mean() < self.bit_depth1.mean():
        #     return self.step_size0

        # if self.weight_q.shape[0] > self.weight_q.shape[1]:
        #     return step_size0

        # stepss = steps0 + steps1
        # stepsm = (2 ** (2 * steps0) + 2 ** (2 * steps1)).multiply(0.5).log2().multiply(0.5)
        # steps2 = stepss.subtract(stepsm)

        return self.step_size1

    @property
    def bias(self):
        return self.linear.bias - ((self.weight_q.float() - self.linear.weight) @ self.offset_q.squeeze()).half() # input_av.half().squeeze()

    def forward_hook_dbg(self, module, layer_in, layer_out) -> None:
        None # print(layer_in[0].float().mean().item())
        # breakpoint()
        # print("forward " + str(self.name))

    def forward_hook(self, module, layer_in, layer_out) -> None:
        self.layer_in = layer_in[0].detach().squeeze()
        input_av = self.layer_in.float() # torch.flatten(self.layer_in, 0, 1)

        self.count_fw += 1
        self.input_av += (1/self.count_fw) * (input_av.mean(0, True) - self.input_av)
        # print(self.count_fw)
        # print("input av", str(self.name), self.input_av.float().mean().item())

    def backward_hook_dbg(self, module, grad_in, grad_out) -> None:
        None # print(grad_out[0].float().mean().item())
        # print("backward " + str(self.name))

    def backward_hook(self, module, grad_in, grad_out) -> None:
        self.grad_out = grad_out[0].detach().squeeze()
        grad_sqr = torch.einsum("ij,ik->jk", self.grad_out, self.layer_in).float()

        self.grad_av0 += (1/self.count_bw) * grad_sqr.mean(1, True)
        self.grad_av1 += (1/self.count_bw) * grad_sqr.mean(0, True)

        self.grad_sq0 += (1/self.count_bw) * grad_sqr.square().mean(1, True)
        self.grad_sq1 += (1/self.count_bw) * grad_sqr.square().mean(0, True)
        self.grad_out = None

        # if self.name == 'model.decoder.layers.11.fc2': #        if self.name == 'model.decoder.layers.11.fc2':
        #     print("backward_hook_fc2", self.layer_in.float().mean().item(), self.grad_sq1.float().mean().item(), grad_out[0].float().mean().item(), grad_in[0].float().mean().item())

        # if self.name == 'model.decoder.layers.11.fc1': #        if self.name == 'model.decoder.layers.11.fc2':
        #     print("backward_hook_fc1", self.layer_in.float().mean().item(), self.grad_sq1.float().mean().item(), grad_out[0].float().mean().item(), grad_in[0].float().mean().item())

        # if self.name == 'model.decoder.layers.10.fc1': #        if self.name == 'model.decoder.layers.11.fc2':
        #     print("backward_hook_fc1", self.grad_sq1.float().mean().item())

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
    def __init__(self, model, layer_ind=0, num_layers=-1, checkpointing=False) -> None:
        super().__init__()
        self.lm_head = model.lm_head #.requires_grad_(False)
        
        if isinstance(model, OPTForCausalLM):
            self.embed_tokens = model.model.decoder.embed_tokens #.requires_grad_(False)
            self.n_head = model.config.num_attention_heads
        if isinstance(model, BloomForCausalLM):
            self.embed_tokens = model.transformer.word_embeddings
            self.n_head = model.config.n_head

        self.model = model.eval().requires_grad_(False)

        # self.num_blocks = len(model.model.de
        # self.blocks = [model.model.decoder.layers[blocksize*i:blocks*(i+1)] for i in range(model.mode)]
        # self.model.model.decoder.layers = self.blocks[layer_ind:layer_ind + num_layers]
        # self.embed_positions = model.model.decoder.embed_positions #.requires_grad_(False)

        model.lm_head = Identity()
        # model.model.decoder.embed_tokens = Identity()

        if checkpointing:
            model.gradient_checkpointing_enable()

        # layers = [(n.split(sep='.'), m) for n, m in model.named_modules() if isinstance(m, LayerNorm)]
        # for layer in layers:
        #     layer[1].requires_grad_(False)

        layers = [(n, m) for n, m in model.named_modules() if n.endswith('query_key_value')]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], Linear3(layer[1], self.n_head))

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], LinearQ(layer[1]))

        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

    def numzero(self):
        zeros = numel = 0
        for i in range(len(self.layers)):
            # breakpoint()
            zeros += self.layers[i][1].bit_depth.count_nonzero()
            numel += self.layers[i][1].bit_depth.numel()

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
            bits += self.layers[i][1].bit_depth.mean().float() * self.layers[i][1].linear.weight.numel()
            numel += self.layers[i][1].linear.weight.numel()

        return bits / numel

    def scale_sq(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].scale_sq()

    def forward(self, input: Tensor) -> Tensor:
        return self.model.forward(inputs_embeds=input)

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
