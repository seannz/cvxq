# import os
import torch

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM #, AutoTokenizer
from optimizer import Optimizer
from options import parse_args
from dataset import wikitext

args = parse_args()
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16)

train_data, valid_data, tests_data = wikitext(model_id=args.model_id)
train_loader = DataLoader(train_data, pin_memory=True)
# valid_loader = DataLoader(train_data, pin_memory=True)
tests_loader = DataLoader(tests_data, pin_memory=True)

optimizer = Optimizer(model, train_loader, tests_loader, args.numbatches, stride=args.stride, pca=args.pca,
                      loglambda=args.log_lambda, bitrate=args.bitrate, checkpointing=args.checkpointing, save_file=args.model_id.replace("/","-") + "-grads-checkpointing.pt")
# breakpoint()
# optimizer.validate()
optimizer.transform()
optimizer.calibrate()
