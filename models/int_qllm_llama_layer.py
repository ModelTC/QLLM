from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig

from models.int_llama_layer import QuantLlamaAttention
from models.transformation import *
from quantize.int_linear import QuantLinear
from quantize.int_linear_lora import LoRAQuantLinear
from quantize.int_matmul import QuantMatMul
from reassembly.cr_module import CRModule


class QLLMLlamaMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        if args.use_lora:
            self.gate_proj = LoRAQuantLinear(
                org_module.gate_proj,
                args.weight_quant_params,
                args.act_quant_params,
                r=args.r,
            )
            self.down_proj = LoRAQuantLinear(
                org_module.down_proj,
                args.weight_quant_params,
                args.act_quant_params,
                r=args.r,
            )
            self.up_proj = LoRAQuantLinear(
                org_module.up_proj,
                args.weight_quant_params,
                args.act_quant_params,
                r=args.r,
            )
        else:
            self.gate_proj = QuantLinear(
                org_module.gate_proj,
                args.weight_quant_params,
                args.act_quant_params,
            )
            self.down_proj = QuantLinear(
                org_module.down_proj,
                args.weight_quant_params,
                args.act_quant_params,
            )
            self.up_proj = QuantLinear(
                org_module.up_proj, args.weight_quant_params, args.act_quant_params
            )
        self.act_fn = ACT2FN[hidden_act]
        self.down_proj_crm = CRModule(self.up_proj, args.channel_ratio)

    def forward(self, x):
        x1 = self.gate_proj(x)
        x1 = self.act_fn(x1)
        x2 = self.up_proj(x)
        x = x1 * x2
        x = self.down_proj_crm(x)
        x = self.down_proj(x)
        return x


class QLLMLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, ori_layer, args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
        )
        self.mlp = QLLMLlamaMLP(
            org_module=ori_layer.mlp,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = ori_layer.input_layernorm
        self.post_attention_layernorm = ori_layer.post_attention_layernorm

        self.input_layernorm_crm = CRModule(self.self_attn, args.channel_ratio)
        self.post_attention_layernorm_crm = CRModule(
            self.mlp.up_proj, args.channel_ratio, self.mlp.act_fn
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.input_layernorm_crm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.post_attention_layernorm_crm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def disassemble_fc(self, num_disassembly, fcs):
        if not isinstance(fcs, list):
            fcs = [fcs]
        for idx, fc in enumerate(fcs):
            fc.weight.data = torch.repeat_interleave(
                fc.weight.data, num_disassembly.int(), dim=1
            )
            fc.in_features = fc.weight.shape[1]

            if isinstance(fc, LoRAQuantLinear):
                fc.lora_A.data = torch.repeat_interleave(
                    fc.lora_A.data, num_disassembly.int(), dim=1
                )

    def assemble_fc(self, src_idx, dst_idx, fcs):
        if not isinstance(fcs, list):
            fcs = [fcs]
        for idx, fc in enumerate(fcs):
            cout, cin = fc.weight.shape
            dev = fc.weight.device
            dtype = fc.weight.dtype
            ori_src_idx = torch.arange(0, cin, 2, device=dev)
            ori_dst_idx = torch.arange(1, cin, 2, device=dev)
            src_idx_ = ori_src_idx[src_idx]
            dst_idx_ = ori_dst_idx[dst_idx]
            r = src_idx_.nelement()

            channel_mask = torch.ones(cin, device=dev, dtype=dtype)
            channel_mask[src_idx_] = 0.0

            src_weight = fc.weight.gather(dim=-1, index=src_idx_.expand(cout, r))
            fc.weight.data.scatter_reduce_(
                dim=-1, index=dst_idx_.expand(cout, r), src=src_weight, reduce="sum"
            )
            fc.weight.data = fc.weight.data.index_select(
                -1, (channel_mask != 0).nonzero().squeeze()
            )
            fc.in_features = (channel_mask != 0).sum()

            if isinstance(fc, LoRAQuantLinear):
                src_weight = fc.lora_A.gather(dim=-1, index=src_idx_.expand(fc.r, r))
                fc.lora_A.data.scatter_reduce_(
                    dim=-1, index=dst_idx_.expand(fc.r, r), src=src_weight, reduce="sum"
                )
                fc.lora_A.data = fc.lora_A.data.index_select(
                    -1, (channel_mask != 0).nonzero().squeeze()
                )

    def output_reassembly(
        self, crm, x, x_max, attention_mask, position_ids, linear_projs, function
    ):
        th = crm.find_threshold(
            x,
            x_max,
            attention_mask,
            position_ids,
            linear_projs,
            function=function,
        )

        crm.find_outlier_channels_and_store(x_max, th)
        if crm.outlier_channel_idx is not None:
            num_additional_channels = (
                crm.num_disassembly.int().sum().item() - linear_projs[0].weight.shape[1]
            )
            crm.num_additional_channels = num_additional_channels
            crm.num_channels = linear_projs[0].weight.shape[1]
            print(f"Disassembling layer {crm.name}")
            print(f"Introduce additional {num_additional_channels} channels")
            self.disassemble_fc(crm.num_disassembly, linear_projs)

            x = crm.disassemble_x(
                x,
                crm.num_disassembly,
                crm.scaling_factors,
            )
            print(f"Assembling layer {crm.name}")
            crm.find_similar_channels_and_store(x, linear_projs)
            self.assemble_fc(crm.src_idx, crm.dst_idx, linear_projs)

    def output_reassembly_with_state_dict(self, crm, linear_projs):
        device = linear_projs[0].weight.device
        crm.outlier_channel_idx.to(device)
        crm.num_disassembly.to(device)
        crm.scaling_factors.to(device)
        crm.src_idx.to(device)
        crm.dst_idx.to(device)
        if crm.outlier_channel_idx is not None:
            num_additional_channels = (
                crm.num_disassembly.int().sum().item() - linear_projs[0].weight.shape[1]
            )
            crm.num_additional_channels = num_additional_channels
            crm.num_channels = linear_projs[0].weight.shape[1]
            print(f"Disassembling layer {crm.name}")
            print(f"Introduce additional {num_additional_channels} channels")
            self.disassemble_fc(crm.num_disassembly, linear_projs)

            print(f"Assembling layer {crm.name}")
            self.assemble_fc(crm.src_idx, crm.dst_idx, linear_projs)
            crm.have_assembled = True

    def input_layernorm_output_reassembly(
        self, x, x_min, x_max, attention_mask, position_ids
    ):
        x_max = torch.maximum(x_min.abs(), x_max)
        self.output_reassembly(
            self.input_layernorm_crm,
            x,
            x_max,
            attention_mask,
            position_ids,
            [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
            function="qkv",
        )

    def input_layernorm_output_reassembly_with_state_dict(self):
        self.output_reassembly_with_state_dict(
            self.input_layernorm_crm,
            [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
        )

    def post_attention_layernorm_output_reassembly(
        self, x, x_min, x_max, attention_mask, position_ids
    ):
        x_max = torch.maximum(x_min.abs(), x_max)
        self.output_reassembly(
            self.post_attention_layernorm_crm,
            x,
            x_max,
            attention_mask,
            position_ids,
            [self.mlp.gate_proj, self.mlp.up_proj],
            function="post_attn_norm",
        )

    def post_attention_layernorm_output_reassembly_with_state_dict(self):
        self.output_reassembly_with_state_dict(
            self.post_attention_layernorm_crm, [self.mlp.gate_proj, self.mlp.up_proj]
        )

    def fc1_output_reassembly(self, x, x_min, x_max, attention_mask, position_ids):
        x_max = torch.maximum(x_min.abs(), x_max)
        self.output_reassembly(
            self.mlp.down_proj_crm,
            x,
            x_max,
            attention_mask,
            position_ids,
            [self.mlp.down_proj],
            function="fc1",
        )

    def fc1_output_output_reassembly_with_state_dict(self):
        self.output_reassembly_with_state_dict(
            self.mlp.down_proj_crm, [self.mlp.down_proj]
        )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, LoRAQuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    def qllm_lora_state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if "lora" in name or "norm" in name:
                destination[prefix + name] = (
                    param if keep_vars else param.detach().cpu()
                )
        return destination

    def qllm_sm_state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_buffers():
            if "crm" in name:
                destination[prefix + name] = (
                    param if keep_vars else param.detach().cpu()
                )
        return destination

    def qllm_state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        destination = self.qllm_lora_state_dict(destination, prefix, keep_vars)
        destination = self.qllm_sm_state_dict(destination, prefix, keep_vars)
        return destination
