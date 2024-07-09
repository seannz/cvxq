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
    parser.add_argument('--groups', dest='groups', type=int, default=4, help='number of quantization groups')
    parser.add_argument('--bitrate', dest='bitrate', type=float, default=3.0, help='bitrate to target')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0, help='error dropout during training')
    parser.add_argument('--train_tokens', dest='train_tokens', type=int, default=1024, help='number of training tokens')
    parser.add_argument('--model_id', dest='model_id', default='facebook/opt-6.7b')
    parser.add_argument('--log_lambda', dest='log_lambda', type=float, default=-30)
    parser.add_argument('--pca', dest='pca', type=int, default=1)
    parser.add_argument('--gpus', dest='gpus', type=int, default=1)
    parser.add_argument('--dtype', dest='dtype', default='torch.float16')
    parser.add_argument('--numbatches', dest='numbatches', type=int, default=140)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=50)
    parser.add_argument('--max_iters', dest='max_iters', type=int, default=1000)
    
    return parser.parse_args()
