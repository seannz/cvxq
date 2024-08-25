import torch

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from huggingface_hub import login
from optimizer import Optimizer
from options import parse_args
from dataset import wikitext, c4

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

args = parse_args()
# login(args.access_token)

# path = hf_hub_download(args.model_id, "model.safetensors.index.json")

# # Create a model and initialize it with empty weights
# config = AutoConfig.from_pretrained(checkpoint)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
# model = load_checkpoint_and_dispatch(model, path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer", "OPTDecoderLayer"])

model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16) #, device_map="auto")

train_data, tests_data1, _ = c4(model_id=args.model_id, train_tokens=args.train_tokens)
_, _, tests_data2 = wikitext(model_id=args.model_id, train_tokens=args.train_tokens)

train_loader = DataLoader(train_data, pin_memory=True)
# valid_loader = DataLoader(train_data, pin_memory=True)
tests_loader1 = DataLoader(tests_data1, pin_memory=True)
tests_loader2 = DataLoader(tests_data2, pin_memory=True)

tests_loaders = [tests_loader1, tests_loader2]

optimizer = Optimizer(model, train_loader, tests_loaders, args.numbatches, args.batch_size, valid_size=args.valid_size, warmup_batches=args.warmup_batches, stride=args.stride, group_size=args.group_size,
                      pca=args.pca, gpus=args.gpus, loglambda=args.log_lambda, bitrate=args.bitrate, max_iters=args.max_iters, checkpointing=args.checkpointing, remarks=args.remarks) #save_file=args.model_id.replace("/","-") + "-quantized")

optimizer.summarize()
optimizer.transform()

optimizer.validate(save_best=False)
optimizer.calibrate()
