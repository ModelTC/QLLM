import torch
import torch.nn as nn


class CDModule(nn.Module):
    def __init__(self, channel_ratio):
        super().__init__()
        self.channel_ratio = channel_ratio
        self.outlier_channel_idx = None
        self.num_disassembly = None
        self.scaling_factors = None
        self.num_additional_channels = 0

    def find_threshold_uniform(self, x_max):
        x_max = x_max.float()
        num_channels = x_max.numel()
        channel_constraint = int(num_channels * self.channel_ratio)
        channelmax_max = x_max.max()
        channelmax_min = x_max.min()

        th = channelmax_max
        step_num = max(100, int(channelmax_max / 0.5))
        step = (channelmax_max - channelmax_min) / step_num
        while th >= channelmax_min:
            num_disassembly = torch.ceil(x_max / th)
            num_disassembly = torch.clamp(num_disassembly, min=1.0)
            num_additional_channels = num_disassembly.int().sum().item() - num_channels
            if num_additional_channels > channel_constraint:
                th += step
                break
            else:
                th -= step
        print("Find threshold {} using uniform method".format(th))
        return th

    def find_outlier_channels(self, x_min, x_max):
        with torch.no_grad():
            x_max = torch.maximum(x_min.abs(), x_max)
            th = self.find_threshold_uniform(x_max)
            outlier_channel_idx = (x_max > th).nonzero().view(-1)
            num_disassembly = torch.ceil(x_max / th)
            num_disassembly = torch.clamp(num_disassembly, min=1.0)
            scaling_factors = (1.0 / num_disassembly).repeat_interleave(num_disassembly.int())
            if len(outlier_channel_idx) != 0:
                del self.outlier_channel_idx
                del self.num_disassembly
                del self.scaling_factors
                self.register_buffer("outlier_channel_idx", outlier_channel_idx)
                self.register_buffer("num_disassembly", num_disassembly)
                self.register_buffer("scaling_factors", scaling_factors)

    def forward(self, x):
        if self.outlier_channel_idx is not None:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            B, N, C = x.shape
            x = x.view(B * N, C)
            x = torch.repeat_interleave(x, self.num_disassembly.int(), dim=1)
            x = x * self.scaling_factors.unsqueeze(0)
            C = x.shape[1]
            x = x.view(B, N, C)
        return x
