import torch
import torch.nn.functional as F
import math

def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x * torch.tanh(F.softplus(x))

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}