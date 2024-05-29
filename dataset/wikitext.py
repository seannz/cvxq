import torch

from datasets import load_dataset
from transformers import AutoTokenizer

class Wikitext:
    def __init__(
            self,
            dataset: str = 'wikitext',
            variant: str = 'wikitext-2-raw-v1',
            split: str = 'train',
            model_id: str = 'facebook/opt-1.3b',
            blocksize: int = 2048,
            numblocks: int = -1,
            **kwargs
    ):
        self.dataset = dataset
        self.variant = variant
        self.split = split
        self.blocksize = blocksize
        self.numblocks = numblocks

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        data = load_dataset(dataset, variant, split=split)
        data = tokenizer('\n\n'.join(data['text']), return_tensors='pt')

        self.data = torch.cat(data.input_ids.split(blocksize, 1)[:numblocks])

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def wikitext(split=None, **kwargs):
    train = Wikitext(split='train', **kwargs)
    valid = Wikitext(split='validation', **kwargs)
    tests = Wikitext(split='test', **kwargs)
    
    return train, valid, tests
