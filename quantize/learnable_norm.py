import torch
import torch.nn as nn

"""
Modify normalization layer to adapt the training of learnable equivalent transformation
"""


class LearnableLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.ori_norm = ori_norm
        self.bias = torch.nn.Parameter(
            torch.zeros(ori_norm.weight.shape, device=ori_norm.weight.device)
        )
        self.variance_epsilon = eps
        self.use_temporary_parameter = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.ori_norm.weight
        bias = self.bias

        return (
            (weight * hidden_states + bias).to(input_dtype)
            if bias is not None
            else (weight * hidden_states).to(input_dtype)
        )
