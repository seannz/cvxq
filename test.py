import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from modules import ModuleQuantized, DeltaOptimizer, OutputAnalyzer
from functorch import make_functional_with_buffers, vmap, grad

model_id = "facebook/opt-125m"
model = ModuleQuantized(AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)).cuda()
fmodel, params, buffers = make_functional_with_buffers(model)
fgrad = torch.func.grad(fmodel)










import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

torch.manual_seed(0);

# Here's a simple CNN and loss function:

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output

def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)

device = 'cuda'

num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)

targets = torch.randint(10, (64,), device=device)

model = SimpleCNN().to(device=device)
predictions = model(data) # move the entire mini-batch through the model

loss = loss_fn(predictions, targets)
loss.backward() # back propogate the 'average' gradient of this mini-batch

def compute_grad(sample, target):
    
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

per_sample_grads = compute_sample_grads(data, targets)

print(per_sample_grads[0].shape)


from functorch import make_functional_with_buffers, vmap, grad

fmodel, params, buffers = make_functional_with_buffers(model)

fmodel

for x in params:
  print(f"{x.shape}")

print(f"\n{type(params)}")

def compute_loss_stateless_model (params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch) 
    loss = loss_fn(predictions, targets)
    return loss

ft_compute_grad = grad(compute_loss_stateless_model)


