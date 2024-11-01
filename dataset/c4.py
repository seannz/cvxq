import torch
import random

from datasets import load_dataset
from transformers import AutoTokenizer

class C4:
    def __init__(
            self,
            dataset: str = 'allenai/c4',
            data_files = {'train': 'en/c4-train.00000-of-01024.json.gz'},
            split: str = 'train',
            model_id: str = 'facebook/opt-1.3b',
            blocksize: int = 2048,
            numblocks: int = 256,
            rand_seed: int = 0,
            **kwargs
    ):
        self.dataset = dataset
        self.split = split
        self.blocksize = blocksize
        self.numblocks = numblocks
        # self.generator = torch.Generator()
        # self.generator.manual_seed(rand_seed)
        self.random = random.Random(rand_seed)

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        data = load_dataset(dataset, split=split, data_files=data_files)

        if split == 'validation':
            data = tokenizer(' '.join(data[:1100]['text']), return_tensors='pt')
            self.data = torch.reshape(data.input_ids[:,:numblocks*blocksize],[numblocks,blocksize])

            return

        encs = []
        found = 0
        for b in range(len(data)):
            if found == numblocks:
                break

            enc = tokenizer(data[b]['text'], return_tensors='pt')

            if enc.input_ids.shape[1] < blocksize:
                continue

            i = 0 # self.random.randint(0, enc.input_ids.shape[1] - blocksize)

            j = i + blocksize # seqlen
            inp = enc.input_ids[:, i:j]
            encs.append(inp)

            found += 1

        self.data = torch.cat(encs) 

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def c4(split=None, train_tokens=256, valid_tokens=256, **kwargs):
    train = C4(split='train', data_files = {'train': 'en/c4-train.00000-of-01024.json.gz'}, numblocks=train_tokens, **kwargs)
    valid = C4(split='validation', data_files = {'validation': 'en/c4-validation.00000-of-00008.json.gz'}, numblocks=valid_tokens, **kwargs)
    # tests = C4(split='test', **kwargs)
    
    return train, valid, None
