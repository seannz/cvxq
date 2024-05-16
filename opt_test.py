import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantizer import ModuleQ
from optimizer import Optimizer

model_ids = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

batchsize = 2 #8
numbatches = 4 #-1
blocksize = 2048
stride = 64
loglambda = -17.15
model_id = model_ids[0]
model = ModuleQ(AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

calibdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
calibdata = tokenizer("\n\n".join(calibdata['text']), return_tensors='pt').input_ids.cuda()
calibdata = torch.cat(calibdata.split(blocksize, 1)[:-1]) #need a DataLoader
calibembd = model.embed_tokens(calibdata).detach().requires_grad_(True)

optimizer = Optimizer(model, calibembd, calibdata, numbatches, batchsize=batchsize, stride=stride,
                      loglambda=loglambda, model_id=model_id, load_file=model_id.replace("/","-") + "-new_gradsq%02d_64_fp16.mat")


# compute the perplexity and bit-rate at a lambda
optimizer.optimize(loglambda)
optimizer.validate()