import gc
import math
import torch
from itertools import cycle
from torch.nn import Module
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from quantizer import ModuleQ

class Optimizer(Module):
    def __init__(self, model, train_loader, tests_loaders, batches, batch_size, stride=1, groups=1, pca=384, loglambda=-18, bitrate=3, max_iters=1000, dropout=0, checkpointing=False, gpus=1, load_file=None, save_file=None) -> None:
        super().__init__()
        self.model = ModuleQ(model, groups=groups, checkpointing=checkpointing, gpus=gpus)
        self.loglambda = loglambda
        self.bitrate = bitrate
        self.train_loader = train_loader
        self.tests_loaders = tests_loaders
        self.batches = batches
        self.batch_size = batch_size
        self.stride = stride
        self.pca = pca
        self.load_file = load_file
        self.save_file = save_file
        self.curr_iter = 0
        self.dropout = dropout
        self.max_iters = max_iters if max_iters is not None else len(self.train_loader)
        # self.optrate = bitrate

        if self.load_file is None: #
            return
        
        self.load_vars()

    def calibrate(self):
        # self.model.add_hooks_dbg()
        for data in cycle(self.train_loader):
            self.model.add_forward_hooks()
            self.model.add_backward_hooks()

            embeds = self.model.embed_tokens(data.to(self.model.device)).requires_grad_(True)
            output = torch.einsum('ij,bkj->bki', self.V, self.model(embeds).logits)
            # output = self.model(embeds).logits
            self.model.remove_forward_hooks()

            self.model.scale_sq()
            for t in tqdm(range(0, output.shape[-2], self.stride)):
                output[:,t, self.curr_iter % self.pca].sum().backward(retain_graph = True)
            del embeds, output

            self.model.remove_backward_hooks()

            self.model.update_offsets()

            self.curr_iter += 1
            if self.curr_iter % self.batch_size == 0:
                self.optimize(self.bitrate, skip=self.curr_iter // self.batch_size % 2, stride=2)
                self.validate()

                # self.optrate -= 0.25 * (self.optrate - self.bitrate) #max(self.bitrate, self.optrate - 0.05)

            # if self.curr_iter % 128 == 0:
            #     self.bitrate -= 0.5
            #     self.clearvar()
            #     # self.quantize_all(self.loglambda, self.dropout)
            #     # self.optimize(self.bitrate, iters=1, dropin=0.01) # + max(0, 1 - self.curr_iter / self.max_iters))

            gc.collect()
            torch.cuda.empty_cache()

        if self.save_file is None:
            return
            
        self.save_vars()

        # self.model.remove_hooks()

    def optimize(self, bitrate, lr=1.9, iters=10, skip=0, stride=1):
        # dual ascent 
        loglambda = self.loglambda #log2_init
        bitrate_curr = bitrate #self.bitrate if bitrate is None else bitrate

        for i in range(iters):
            loglambda += lr * (bitrate_curr - bitrate)
            # self.quantize_all(loglambda, skip, stride)
            self.optimize_all(loglambda)
            bitrate_curr = self.model.bitrate()
            # print("optimize iteration %02d, loglambda: %f, bitrate: %f," % (i, loglambda, bitrate_curr))

        self.quantize_all(loglambda, skip, stride)
        self.loglambda = loglambda # log2_lambda
        self.model.update_offsets()

        return loglambda, bitrate_curr

    def bitgroup_all(self):
        self.model.bitgroup_all()

    def clearvar(self):
        self.model.clearvar_all()

    def transform(self):
        means = 0
        covars = 0

        with torch.no_grad():
            for data in tqdm(self.train_loader):
                embeds = self.model.embed_tokens(data.to(self.model.device))
                output = self.model(embeds).logits.float()
                means += output.sum([-3,-2]) * (1 / self.train_loader.dataset.blocksize)
                # output *= math.sqrt(1 / self.train_loader.dataset.blocksize)
                covars += torch.einsum('bji,bjk->ik', output, output).multiply_(1 / self.train_loader.dataset.blocksize)

        covars -= means.reshape(-1,1) @ means.reshape(1,-1)

        self.V = torch.linalg.eig(covars)[1].T[:self.pca].real.to(embeds.dtype)

    def validate(self):
        ppls = []
        for tests_loader in self.tests_loaders:
            with torch.no_grad():
                losses = 0
                for data in tests_loader: #self.tests_loaders[0]:
                    embeds = self.model.embed_tokens(data.to(self.model.device))
                    logits = self.model.lm_head(self.model(embeds).logits)[:,:-1].flatten(0,1)
                    labels = data[:,1:].to(logits.device).flatten(0,1)
                    losses += cross_entropy(logits.float(), labels, reduction='none').sum()

                losses = losses / len(tests_loader.dataset) / (tests_loader.dataset.blocksize - 1)
                ppls = ppls + [torch.exp(losses).item()]

        bit = self.model.bitrate()
        lam = self.model.log2lam()
        zer = self.model.numzero()
        print("iterations: %03d, perplexity: %f, %f, lambda: %f, bitrate: %f, numzero: %f, dropout: %f" % (self.curr_iter, ppls[0], ppls[1], lam, bit, zer, self.dropout))
        # breakpoint()

        return ppls, bit, zer

    def summarize(self):
        print(self.model)

    def unquantize_all(self):
        self.model.unquantize_all()

    def optimize_all(self, log2_lambda):
        self.model.optimize_all(log2_lambda)

    def quantize_all(self, log2_lambda, skip=0, stride=1): #, batches=None):
        # batches = self.batches if batches is None else batches
        # log2_lambda = self.loglambda if log2_lambda is None else log2_lambda
        self.model.quantize_all(log2_lambda, skip=skip, stride=stride) # + math.log2(batches))

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
