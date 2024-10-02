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
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

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
    def __init__(self, linear, group_size=-1, name=None) -> None:
        super().__init__()
        self.linear = linear
        self.cloned = copy.deepcopy(linear)

        if self.cloned.bias is None:
            self.cloned.bias = Parameter(torch.zeros_like(linear.weight[:,0]), requires_grad=linear.weight.requires_grad)

        self.count_fw = 0
        self.count_bw = 0

        self.groups = groups = max(linear.weight.shape[0] // group_size, 1)
        self.group_size = group_size
        self.name = name

        self.register_buffer('input_av', torch.zeros([1, linear.weight.shape[1]], dtype=linear.weight.dtype))
        self.register_buffer('grad_sq0', torch.zeros([linear.weight.shape[0], 1], dtype=linear.weight.dtype))
        self.register_buffer('grad_sq2', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.weight.dtype))

        self.register_buffer('groups_2', torch.arange(linear.weight.shape[0]).reshape([groups, -1]))
        self.register_buffer('bit_depth2', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.weight.dtype))

        # self.register_buffer('weight', linear.weight.detach())
        # self.register_buffer('offset', linear.bias.detach() if linear.bias is not None else torch.zeros_like(linear.weight[:,0].detach()))
        # self.register_buffer('mult', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.weight.dtype))
        # self.register_buffer('bias', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.weight.dtype))
        # self.register_buffer('best_groups_2', torch.arange(linear.weight.shape[0]).reshape([groups, -1]))
        # self.register_buffer('best_bit_depth2', torch.zeros([groups, 1, linear.weight.shape[1]], dtype=linear.weight.dtype))
        

    def forward(self, input: Tensor) -> Tensor:
        # breakpoint()
        return self.cloned(input) #F.linear(input, self.weight, self.offset)
        # return output

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

    def optimize(self, regroup=False):
        groups = self.bit_group() if regroup else self.groups_2
        weight = self.linear.weight[groups] #.float()
        offset = torch.median(weight, 1, keepdim=True).values
        # offset = torch.mean(weight, 1, keepdim=True)
        bit_depth = self.bit_depth((weight - offset).float()) #.float()

        self.groups_2.copy_(groups.to(self.groups_2.dtype))
        self.bit_depth2.copy_(bit_depth.to(self.bit_depth2.dtype))

    def quantize(self, grid_search=False):
        groups = self.groups_2 #.bit_group()
        weight = self.linear.weight[groups]
        offset = torch.median(weight, 1, keepdim=True).values
        #offset = torch.mean(weight, 1, keepdim=True) #.values

        # self.count_qt += 1
        # bit_depth = self.bit_depth1.add_((1/self.count_qt) * (self.bit_depth2 - self.bit_depth1)).round().float()
        bit_depth = self.bit_depth2.round().float() #1.add_((1/self.count_qt) * (self.bit_depth2 - self.bit_depth1)).round().float()
        step_size = self.step_size((weight - offset).float())

        best_bias = bias = torch.zeros_like(offset)
        best_mult = mult = torch.zeros_like(offset)
        best_msqe = torch.zeros_like(offset).fill_(torch.inf)
        
        if grid_search:
            for mult in torch.arange(-2,2.125,0.125):
                quants = Laplace(offset + bias * 2. ** step_size, (3/1.414213562) * 2. ** (step_size + mult)).cdf(weight).multiply_(2. ** bit_depth) # * (1 / (1 - 2 * quantiles)))
                quants = quants.floor_().clamp_(None, 2. ** bit_depth - 1).add_(0.5)
                quants = Laplace(offset + bias * 2. ** step_size, (3/1.414213562) * 2. ** (step_size + mult)).icdf(quants.multiply_(2. **-bit_depth)) #.multiply_(2. **-bit_depth)) # * (1 - 2 * quantiles)))

                msqe = torch.mean(((quants - weight).abs_() ** 2), 1, keepdim=True, dtype=torch.float)
                best_mult = best_mult.where(best_msqe < msqe, mult)
                best_msqe = best_msqe.where(best_msqe < msqe, msqe)

            mult = best_mult

            for bias in torch.arange(-2,2.125,0.125):
                quants = Laplace(offset + bias * 2. ** (step_size + mult), (3/1.414213562) * 2. ** (step_size + mult)).cdf(weight).multiply_(2. ** bit_depth) # * (1 / (1 - 2 * quantiles)))
                quants = quants.floor_().clamp_(None, 2. ** bit_depth - 1).add_(0.5)
                quants = Laplace(offset + bias * 2. ** (step_size + mult), (3/1.414213562) * 2. ** (step_size + mult)).icdf(quants.multiply_(2. **-bit_depth)) #.multiply_(2. **-bit_depth)) # * (1 - 2 * quantiles)))
                
                msqe = torch.mean(((quants - weight).abs_() ** 2), 1, keepdim=True, dtype=torch.float)
                best_bias = best_bias.where(best_msqe < msqe, bias)
                best_msqe = best_msqe.where(best_msqe < msqe, msqe)
                
            bias = best_bias
        # else:
        #     mult = self.mult
        #     bias = self.bias

        quants = Laplace(offset + bias * 2. ** (step_size + mult), (3/1.414213562) * 2. ** (step_size + mult)).cdf(weight).multiply_(2. ** bit_depth) # * (1 / (1 - 2 * quantiles)))
        quants = quants.floor_().clamp_(None, 2. ** bit_depth - 1).add_(0.5)
        quants = Laplace(offset + bias * 2. ** (step_size + mult), (3/1.414213562) * 2. ** (step_size + mult)).icdf(quants.multiply_(2. **-bit_depth)) #.multiply_(2. **-bit_depth)) # * (1 - 2 * quantiles)))

        # self.bias.copy_(bias)
        # self.mult.copy_(mult)

        self.cloned.weight[groups] = quants.to(self.cloned.weight.dtype)
        # print("best step multiplier: %f" % mult)
        # introduce error randomization
        # if rand_p > 0: #dropout > 0:
        #     rands = torch.rand_like(self.weight).lt_(rand_p).multiply_(2)
        #     self.weight.add_(rands.multiply_(self.linear.weight - self.weight))

        if quants.isnan().any():
            breakpoint()
        
        # self.groups_2.copy_(groups.to(self.groups_2.dtype))
        # self.bit_depth2.copy_(bit_depth.to(self.bit_depth2.dtype))

    def unquantize(self):
        self.cloned.weight.copy_(self.linear.weight)

    def bit_group(self):
        bit_group = torch.var(self.linear.weight.float(), 1, keepdim=True).log2_().add_(self.grad_sq0.float().log2())
        bit_group = torch.stack(bit_group.squeeze(1).sort(0).indices.tensor_split(self.groups)) #.sort(1).values

        return bit_group

    def bit_depth(self, weight):
        bit_depth = torch.mean(weight ** 2, 1, keepdim=True).float().log2_().add_(self.grad_sq2.float().log2()).subtract_(self.log2_lambda).multiply_(0.5)
        # bit_depth = torch.mean(weight ** 2, 1, keepdim=True, dtype=torch.float).log2_().add_(self.grad_sq2.float().log2()).subtract_(self.log2_lambda).multiply_(0.5)
        bit_depth = torch.clamp_(bit_depth, 0, 8) #.round_()
        # bit_depth = torch.threshold_(bit_depth, 0.0, 0).round_() #.clamp_(0, 8) #.to(weight.dtype)

        return bit_depth

    def step_size(self, weight):
        # step_sizp = torch.sum(weight ** 2 * weight.gt(0), 1, keepdim=True, dtype=torch.float).divide_(torch.sum( weight.gt(0), 1, keepdim=True, dtype=torch.float))
        # step_sizm = torch.sum(weight ** 2 *~weight.gt(0), 1, keepdim=True, dtype=torch.float).divide_(torch.sum(~weight.gt(0), 1, keepdim=True, dtype=torch.float))
        # step_size = torch.where(weight.gt(0), step_sizp, step_sizm).log2_().multiply_(0.5)

        step_size = torch.mean(weight ** 2, 1, keepdim=True, dtype=torch.float).log2_().multiply_(0.5)  #.divide_(torch.sum( weight.gt(0), 1, keepdim=True, dtype=torch.float))
        # step_size = torch.median(weight ** 2, 1, keepdim=True).values.log2_().multiply_(0.5)  #.divide_(torch.sum( weight.gt(0), 1, keepdim=True, dtype=torch.float))

        return step_size #.where(step_size > 0, 1e-5) #p, step_sizm

    def forward_hook(self, module, layer_in, layer_out) -> None:
        self.layer_in = layer_in[0].detach().squeeze()
        input_av = self.layer_in # torch.flatten(self.layer_in, 0, 1)

        self.count_fw += 1
        self.input_av += (1/(self.count_fw)) * (input_av.mean(0, True, dtype=torch.float).subtract_(self.input_av))

    def backward_hook(self, module, grad_in, grad_out) -> None:
        self.grad_out = grad_out[0].detach().nan_to_num_().squeeze()
        grad_sqr = torch.einsum("ij,ik->jk", self.grad_out, self.layer_in).float().square()

        # breakpoint()
        self.grad_sq0 += (1/(self.count_bw)) * grad_sqr.mean(1, True, dtype=torch.float)
        # self.grad_sq1 += (1/self.count_bw) * grad_sqr.mean(0, True, dtype=torch.float)
        self.grad_sq2 += (1/(self.count_bw)) * grad_sqr[self.groups_2].mean(1, True, dtype=torch.float)

        # replace nan's with the current running averages
        # self.grad_out.where(self.grad_out.isnan().any(1, keepdim=True), 0)

        if self.grad_out.isnan().any():
            breakpoint()

        # if grad_in[0].isnan().any():
        #     breakpoint()

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
        bias = self.linear.bias if self.linear.bias is not None else 0
        self.cloned.bias.copy_(bias - ((self.cloned.weight - self.linear.weight) @ self.input_av.squeeze()).to(self.linear.weight.dtype)) #.squeeze())

    def update_best(self) -> None:
        self.best_groups_2.copy_(self.groups_2)
        self.best_bit_depth2.copy_(self.bit_depth2)

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

class OPTBlockQ(Module):
    def __init__(self, block, name=None) -> None:
        super().__init__()
        self.block = block

    def forward(self, input, attention_mask, *args, **kwargs) -> Tensor:
        device = self.block.self_attn.q_proj.cloned.weight.device
        output = self.block(input.to(device), attention_mask.to(device) if attention_mask is not None else None, *args, **kwargs)

        return output

class LlamaBlockQ(Module):
    def __init__(self, block, name=None) -> None:
        super().__init__()
        self.block = block

    def forward(self, input, attention_mask, position_ids, *args, **kwargs) -> Tensor:
        device = self.block.self_attn.q_proj.cloned.weight.device
        output = self.block(input.to(device), attention_mask.to(device) if attention_mask is not None else None,
                            position_ids.to(device) if position_ids is not None else None, *args, **kwargs)
        return output

class ModuleQ(Module):
    def create(model, group_size=-1, checkpointing=False, gpus=1):
        # path = hf_hub_download(model_id, "model.safetensors.index.json")
        # # Create a model and initialize it with empty weights
        # config = AutoConfig.from_pretrained(checkpoint)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

        if isinstance(model, OPTForCausalLM):
            return OPTQ(model, group_size, checkpointing, gpus)
        if isinstance(model, LlamaForCausalLM):
            return LlamaQ(model, group_size, checkpointing, gpus)

        # model = load_checkpoint_and_dispatch(model, path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer", "OPTDecoderLayer"])

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

    def optimize_all(self, log2_lambda, regroup=False) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].log2_lambda = log2_lambda
            self.layers[i][1].optimize(regroup=regroup) # = True

    def update_best_all(self) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].update_best() # = True

    def quantize_all(self, log2_lambda, grid_search=False) -> None:
        for i in range(len(self.layers)):
            self.layers[i][1].quantize(grid_search=grid_search) # = True

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


class LlamaQ(ModuleQ):
    def __init__(self, model, group_size=-1, checkpointing=False, gpus=1) -> None:
        super().__init__()
        self.embed_tokens = model.get_input_embeddings() #.model.embed_tokens #.requires_grad_(False)
        self.embed_dimens = model.config.hidden_size # // groups
        self.group_size = group_size
        self.model = model.eval().requires_grad_(False)
        self.lm_head = copy.deepcopy(model.lm_head).cuda(gpus - 1) #.requires_grad_(False)
        model.lm_head = Identity()

        if checkpointing:
            model.gradient_checkpointing_enable()

        layers = [(n, m) for n, m in model.model.layers.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model.model.layers), layer[0].split('.')[-1], LinearQ(layer[1], group_size))
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LlamaDecoderLayer)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], LlamaBlockQ(layer[1]))
        self.blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, LlamaBlockQ)]

        model.model.embed_tokens.cuda(0)
        for i in range(len(model.model.layers)):
            model.model.layers[i].cuda(int(i / len(model.model.layers) * gpus))
        model.model.norm.cuda(gpus - 1)

    def save_quantized(self, save_directory):
        model = self.model
        model.lm_head = self.lm_head

        layers = [(n, m) for n, m in model.get_decoder().named_modules() if isinstance(m, LinearQ)]
        blocks = [(n, m) for n, m in model.get_decoder().named_modules() if isinstance(m, LlamaBlockQ)]

        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model.get_decoder()), layer[0].split('.')[-1], layer[1].cloned)
        for block in blocks:
            setattr(reduce(getattr, block[0].split('.')[:-1], model.get_decoder()), block[0].split('.')[-1], block[1].block)
        model.save_pretrained(save_directory)

        for block in blocks:
            setattr(reduce(getattr, block[0].split('.')[:-1], model.get_decoder()), block[0].split('.')[-1], block[1])
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model.get_decoder()), layer[0].split('.')[-1], layer[1])
        model.lm_head = Identity()        

class OPTQ(ModuleQ):
    def __init__(self, model, group_size=-1, checkpointing=False, gpus=1) -> None:
        super().__init__()
        self.embed_tokens = model.get_input_embeddings() # model.model.decoder.embed_tokens
        self.embed_dimens = model.config.hidden_size
        self.group_size = group_size
        self.model = model.eval().requires_grad_(False)
        self.lm_head = copy.deepcopy(model.lm_head).cuda(gpus - 1)
        model.lm_head = Identity()

        if checkpointing:
            model.gradient_checkpointing_enable()

        layers = [(n, m) for n, m in model.model.decoder.layers.named_modules() if isinstance(m, Linear)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model.model.decoder.layers), layer[0].split('.')[-1], LinearQ(layer[1], group_size))
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LinearQ)]

        layers = [(n, m) for n, m in model.named_modules() if isinstance(m, OPTDecoderLayer)]
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model), layer[0].split('.')[-1], OPTBlockQ(layer[1]))
        self.blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, OPTBlockQ)]

        model.model.decoder.embed_tokens.cuda(0)
        model.model.decoder.embed_positions.cuda(0)
        if model.model.decoder.project_in is not None:
            model.model.decoder.project_in.cuda(0)
        for i in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[i].cuda(int(i / len(model.model.decoder.layers) * gpus))
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out.cuda(gpus - 1)
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm.cuda(gpus - 1)

    def save_quantized(self, save_directory):
        model = self.model
        model.lm_head = self.lm_head

        layers = [(n, m) for n, m in model.get_decoder().named_modules() if isinstance(m, LinearQ)]
        blocks = [(n, m) for n, m in model.get_decoder().named_modules() if isinstance(m, OPTBlockQ)]

        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model.get_decoder()), layer[0].split('.')[-1], layer[1].cloned)
        for block in blocks:
            setattr(reduce(getattr, block[0].split('.')[:-1], model.get_decoder()), block[0].split('.')[-1], block[1].block)
        model.save_pretrained(save_directory)

        for block in blocks:
            setattr(reduce(getattr, block[0].split('.')[:-1], model.get_decoder()), block[0].split('.')[-1], block[1])
        for layer in layers:
            setattr(reduce(getattr, layer[0].split('.')[:-1], model.get_decoder()), layer[0].split('.')[-1], layer[1])
        model.lm_head = Identity()        




