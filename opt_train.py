# import os
import torch

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM #, AutoTokenizer
from optimizer import Optimizer
from options import parse_args
from dataset import wikitext, c4

args = parse_args()
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16)

train_data, tests_data1, _ = c4(model_id=args.model_id, train_tokens=args.train_tokens)
_, _, tests_data2 = wikitext(model_id=args.model_id, train_tokens=args.train_tokens)

train_loader = DataLoader(train_data, pin_memory=True)
# valid_loader = DataLoader(train_data, pin_memory=True)
tests_loader1 = DataLoader(tests_data1, pin_memory=True)
tests_loader2 = DataLoader(tests_data2, pin_memory=True)

tests_loaders = [tests_loader1, tests_loader2]

optimizer = Optimizer(model, train_loader, tests_loaders, args.numbatches, args.batch_size, stride=args.stride, groups=args.groups, pca=args.pca, gpus=args.gpus,
                      loglambda=args.log_lambda, bitrate=args.bitrate, dropout=args.dropout, checkpointing=args.checkpointing, save_file=args.model_id.replace("/","-") + "-grads-checkpointing.pt")

optimizer.summarize()
optimizer.transform()

optimizer.validate()
optimizer.calibrate()
