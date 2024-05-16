import torch
# from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantizer import ModuleQ
from optimizer import Optimizer #, OutputAnalyzer

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

batchsize = 1 #8
numblocks = 8 #-1
blocksize = 2048
model_id = model_ids[0]
model = ModuleQ(AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

calibdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
calibdata = tokenizer("\n\n".join(calibdata['text']), return_tensors='pt').input_ids.cuda()
calibdata = torch.cat(calibdata.split(blocksize, 1)[:numblocks]) #need a DataLoader

optimizer = Optimizer(model, calibdata, batchsize=batchsize, model_id=model_id, load_file="inputs%02d_16_fp16.mat")
optimizer.calibrate(backward=True)
optimizer.optimize(log2lam=-17)
ppl = optimizer.validate()
bit = model.bitrate()

print("perplexity: %f" % ppl)
print("bitrate: %f" % bit)
