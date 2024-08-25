from argparse import ArgumentParser

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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpointing', dest='checkpointing', action='store_true')
    parser.add_argument('--stride', dest='stride', type=int, default=64, help='stride for tokens')
    parser.add_argument('--group_size', dest='group_size', type=int, default=512, help='size of quantization groups')
    parser.add_argument('--bitrate', dest='bitrate', type=float, default=3.0, help='bitrate to target')
    parser.add_argument('--train_tokens', dest='train_tokens', type=int, default=1024, help='number of training tokens')
    parser.add_argument('--model_id', dest='model_id', default='facebook/opt-6.7b')
    parser.add_argument('--log_lambda', dest='log_lambda', type=float, default=-30)
    parser.add_argument('--pca', dest='pca', type=int, default=1)
    parser.add_argument('--gpus', dest='gpus', type=int, default=1)
    parser.add_argument('--dtype', dest='dtype', default='torch.float16')
    parser.add_argument('--numbatches', dest='numbatches', type=int, default=140)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('--warmup_batches', dest='warmup_batches', type=int, default=2)
    parser.add_argument('--valid_size', dest='valid_size', type=int, default=16)
    parser.add_argument('--max_iters', dest='max_iters', type=int, default=256)
    parser.add_argument('--remarks', dest='remarks', default='temp') #, type=int, default=256)
    parser.add_argument('--access_token', dest='access_token', default='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')
    return parser.parse_args()
