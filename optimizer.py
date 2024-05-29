import math
import torch

from torch.nn import Module
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from quantizer import ModuleQ

class Optimizer(Module):
    def __init__(self, model, train_loader, tests_loader, batches, stride=1, pca=384, loglambda=-18, bitrate=3, checkpointing=False, load_file=None, save_file=None) -> None:
        super().__init__()
        self.model = ModuleQ(model, checkpointing=checkpointing).cuda()
        self.loglambda = loglambda
        self.bitrate = bitrate
        self.train_loader = train_loader
        self.tests_loader = tests_loader
        self.batches = batches
        self.stride = stride
        self.pca = pca
        self.load_file = load_file
        self.save_file = save_file
        self.curr_iter = 0

        if self.load_file is None: #
            return
        
        self.load_vars()

    def calibrate(self):
        # self.model.add_hooks_dbg()
        for data in self.train_loader:
            self.model.add_forward_hooks()
            self.model.add_backward_hooks()
            embeds = self.model.embed_tokens(data.to(self.model.device)).requires_grad_(True)
            output = torch.einsum('ij,bkj->bki', self.V, self.model(embeds).logits)
            self.model.remove_forward_hooks()

            self.model.scale_sq()
            for t in tqdm(range(0, output.shape[-2], self.stride)):
                # for s in range(output.shape[-1]):
                output[:,t, self.curr_iter % self.pca].sum().backward(retain_graph = True)
            del embeds, output
            self.model.remove_backward_hooks()

            self.model.update_offsets()

            # breakpoint()
            if self.curr_iter % 1 == 0: 
                self.optimize()

            if self.curr_iter % 10 == 0:
                self.validate()

            self.curr_iter += 1

        if self.save_file is None:
            return
            
        self.save_vars()

        # self.model.remove_hooks()

    def optimize(self, bitrate=None, lr=0.95, iters=10):
        # dual ascent 
        bitrate = self.bitrate if bitrate is None else bitrate
        loglambda = self.loglambda #log2_init

        for i in range(iters):
            self.quantize_all(loglambda)
            bitrate_curr = self.model.bitrate()
            loglambda += lr * (bitrate_curr - bitrate)

        self.loglambda = loglambda # log2_lambda

        return loglambda, bitrate_curr

    def transform(self):
        means = 0
        covars = 0

        with torch.no_grad():
            for data in tqdm(self.train_loader):
                embeds = self.model.embed_tokens(data.to(self.model.device))
                output = self.model(embeds).logits.float()
                means += output.sum([-3,-2]) * (1 / self.train_loader.dataset.blocksize)
                # output *= math.sqrt(1 / self.train_loader.dataset.blocksize)
                covars += torch.einsum('bji,bjk->ik', output, output) * (1 / self.train_loader.dataset.blocksize)

        covars -= means.reshape(-1,1) @ means.reshape(1,-1)
        # covars *= (1 / len(self.train_loader.dataset))
        self.V = torch.linalg.eig(covars)[1].T[:self.pca].real.half()

    def validate(self):
        losses = 0
        with torch.no_grad():
            for data in self.tests_loader:
                embeds = self.model.embed_tokens(data.to(self.model.device))
                logits = self.model.lm_head(self.model(embeds).logits)[:,:-1].flatten(0,1)
                labels = data[:,1:].to(self.model.device).flatten(0,1)
                losses += cross_entropy(logits.float(), labels, reduction='none').sum()

        losses = losses / len(self.tests_loader.dataset) / (self.tests_loader.dataset.blocksize - 1)

        ppl = torch.exp(losses)
        bit = self.model.bitrate()
        lam = self.model.log2lam()
        zer = self.model.numzero()
        print("iterations: %03d, perplexity: %f, lambda: %f, bitrate: %f, numzero: %f" % (self.curr_iter, ppl, lam, bit, zer))

        return ppl, bit, zer

    def unquantize_all(self):
        self.model.unquantize_all()

    def quantize_all(self, log2_lambda=None, batches=None):
        batches = self.batches if batches is None else batches
        log2_lambda = self.loglambda if log2_lambda is None else log2_lambda

        self.model.quantize_all(log2_lambda + math.log2(batches))

    def load_vars(self):
        data = torch.load(self.load_file)
        for i in range(len(self.model.layers)):
            # self.model.layers[i][1].grad_sq0 = data["%02d_grad_sq0" % i].to(self.device)
            self.model.layers[i][1].grad_sq1 = data["%02d_grad_sq1" % i].to(self.device)
            # self.model.layers[i][1].weight_0 = data["%02d_weight_0" % i].to(self.device)
            self.model.layers[i][1].weight_1 = data["%02d_weight_1" % i].to(self.device)

    def save_vars(self, data={}):
        for i in range(len(self.model.layers)):
            # data["%02d_grad_sq0" % i] = self.model.layers[i][1].grad_sq0.detach_().cpu()
            data["%02d_grad_sq1" % i] = self.model.layers[i][1].grad_sq1.detach_().cpu()
            # data["%02d_weight_0" % i] = self.model.layers[i][1].weight_0.detach_().cpu()
            data["%02d_weight_1" % i] = self.model.layers[i][1].weight_1.detach_().cpu()
        torch.save(data, self.save_file)

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device
