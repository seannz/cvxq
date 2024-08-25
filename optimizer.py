import gc
import math
import torch
from itertools import cycle
from torch.nn import Module
from torch.nn.functional import cross_entropy
from accelerate import Accelerator
from tqdm import tqdm

from quantizer import ModuleQ

class Optimizer(Module):
    def __init__(self, model, train_loader, tests_loaders, batches, batch_size, valid_size=16, stride=1, group_size=-1, warmup_batches=2, pca=384, loglambda=-18, bitrate=3, max_iters=1000, rand_p=0.5, checkpointing=False, gpus=1, remarks="temp", load_file=None, save_file=None) -> None:
        super().__init__()
        self.model = ModuleQ.create(model, group_size=group_size, checkpointing=checkpointing, gpus=gpus)
        self.loglambda = loglambda
        self.bitrate = bitrate
        self.train_loader = train_loader
        self.tests_loaders = tests_loaders
        self.batches = batches
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.stride = stride
        self.pca = pca
        self.load_file = load_file
        self.save_file = save_file
        self.curr_iter = 0 # -warmup_batches * batch_size
        self.max_iters = max_iters # if max_iters is not None else len(self.train_loader)
        self.best_ppls = [torch.inf] * len(tests_loaders)
        self.pca_index = 0
        self.pca_reset_after = 10
        self.pca_reset_tol = 2e-2
        self.accelerator = Accelerator()
        self.remarks = remarks

        # if self.load_file is None: #
        #     return
        
        # self.load_vars()

    def calibrate(self):
        for data in cycle(self.train_loader):
            self.model.add_forward_hooks()
            self.model.add_backward_hooks()

            embeds = self.model.embed_tokens(data.to(self.model.device)).requires_grad_(True)
            output = torch.einsum('ij,bkj->bki', self.V, self.model(embeds).logits.to(embeds.dtype)) # - logits.mean(2, keepdim=True))

            self.model.remove_forward_hooks()

            self.model.scale_sq()
            # for t in tqdm(range(0, output.shape[-2] + 1, self.stride)):
            for t in tqdm(torch.arange(0, output.shape[-2], self.stride - self.stride / output.shape[-2]).int()):
                # self.accelerator.backward(output[:, t, self.pca_index % min(output.shape[-1], self.pca)].sum(), retain_graph = True)
                output[:, t, self.pca_index % min(output.shape[-1], self.pca)].sum().backward(retain_graph = True)
            del embeds, output

            self.model.remove_backward_hooks()
            self.model.update_offsets()

            self.pca_index += 1
            self.curr_iter += 1

            # if self.curr_iter >= 0:
            if self.curr_iter % self.batch_size == 0:
                self.optimize(self.bitrate, regroup=True) #self.curr_iter % 1 * self.batch_size == 0)

                if self.curr_iter % self.valid_size == 0:
                    self.quantize_all(self.loglambda, grid_search=True) #self.curr_iter > 2 * self.batch_size) #, rand_p=self.rand_p)
                    self.model.update_offsets()
                    self.validate()

                self.quantize_all(self.loglambda, grid_search=False) #self.curr_iter > 2 * self.batch_size) #, rand_p=self.rand_p)
                self.model.update_offsets()

                # self.quantize_all(self.loglambda) # , rand_p=self.rand_p)
                # self.model.update_offsets()
            gc.collect()
            torch.cuda.empty_cache()

            if self.curr_iter >= self.max_iters:
                break

        # if self.save_file is None:
        #     return
            
        # self.save_vars()

    def optimize(self, bitrate, regroup=False, lr=1.9, iters=10):
        # dual ascent 
        loglambda = self.loglambda #log2_init
        bitrate_curr = bitrate #self.bitrate if bitrate is None else bitrate

        for i in range(iters):
            loglambda += lr * (bitrate_curr - bitrate)
            # self.quantize_all(loglambda, skip, stride)
            self.optimize_all(loglambda, regroup=regroup)
            bitrate_curr = self.model.bitrate()

        # self.quantize_all(loglambda)
        self.loglambda = loglambda # log2_lambda
        # self.model.update_offsets()

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
        eigenv = torch.linalg.eig(covars)[1].T[:self.pca].real #.to(embeds.dtype)
        # offset = torch.ones_like(eigenv[0:1]) * (1 / math.sqrt(eigenv[0].numel()))

        self.V = torch.cat([eigenv]).to(embeds.dtype).contiguous()

    def validate(self, save_best=True):
        # loglambda = self.loglambda
        # self.quantize_all(loglambda)
        # self.model.update_offsets()

        ppls = []
        for tests_loader in self.tests_loaders:
            with torch.no_grad():
                losses = 0
                for data in tests_loader: #self.tests_loaders[0]:
                    embeds = self.model.embed_tokens(data.to(self.model.device))
                    logits = self.model.lm_head(self.model(embeds).logits.to(embeds.dtype))[:,:-1].flatten(0,1)
                    labels = data[:,1:].to(logits.device).flatten(0,1)
                    losses += cross_entropy(logits.float(), labels, reduction='none').sum()

                losses = losses / len(tests_loader.dataset) / (tests_loader.dataset.blocksize - 1)
                ppls = ppls + [torch.exp(losses).item()]

        bit = self.model.bitrate()
        lam = self.model.log2lam()
        zer = self.model.numzero()
        print("iterations: %03d, perplexity: %f, %f, lambda: %f, bitrate: %7.4f, numzero: %f" % (self.curr_iter, ppls[0], ppls[1], lam, bit, zer))

        if not save_best:
            return

        self.save_quantized_model(ppls)

    def save_quantized_model(self, ppls):
        if ppls[0] < self.best_ppls[0]:
            self.best_ppls[:] = ppls[:]
            self.model.save_quantized(self.remarks)

    def reset_on_plateau(self, ppls):
        if self.curr_iter == 0:
            return

        if self.best_ppls[0] > ppls[0]:
            self.best_iter = self.curr_iter
            self.best_ppls[:] = ppls[:]

        if self.best_ppls[0] - self.pca_reset_tol < ppls[0] and self.curr_iter - self.best_iter >= self.pca_reset_after * self.batch_size:
            self.pca_index = 0
            self.pca_reset_after *= 2
            self.best_iter = self.curr_iter
            self.best_ppls[:] = ppls[:]
            print("resetting pca index to 0")
            
    def summarize(self):
        print(self.model)

    def unquantize_all(self):
        self.model.unquantize_all()

    def optimize_all(self, log2_lambda, regroup=False):
        self.model.optimize_all(log2_lambda, regroup=regroup)

    def quantize_all(self, log2_lambda, grid_search=False): #, batches=None):
        # batches = self.batches if batches is None else batches
        # log2_lambda = self.loglambda if log2_lambda is None else log2_lambda
        self.model.quantize_all(log2_lambda, grid_search=grid_search) # + math.log2(batches))

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

        torch.save(data, self.save_file)

    # @property
    # def config(self):
    #     return self.model.config

    # @property
    # def device(self):
    #     return self.model.device
