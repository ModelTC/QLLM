import math

import torch
import torch.nn as nn

from quantize.int_linear import QuantLinear


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRAQuantLinear(QuantLinear, LoRALayer):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
    ):
        super().__init__(
            org_module, weight_quant_params, act_quant_params, disable_input_quant
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        if r > 0:
            out_features, in_features = self.weight.shape
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        if self.r > 0 and not self.merged and self.use_weight_quant:
            out = self.fwd_func(
                input,
                weight + self.lora_B @ self.lora_A * self.scaling,
                bias,
                **self.fwd_kwargs
            )
        else:
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def extra_repr(self):
        s = super().extra_repr()
        s += ", use_act_quant={}".format(self.use_act_quant)
        s += ", use_weight_quant={}".format(self.use_weight_quant)
        s += ", disable_input_quant={}".format(self.disable_input_quant)
        s += ", lora_quant"
        return s
