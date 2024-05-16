import torch
import scipy
import math
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module, Linear, Identity
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import reduce
from transformers.models.opt.modeling_opt import OPTDecoderLayer 
from torch.distributions.normal import Normal

