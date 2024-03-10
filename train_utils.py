from collections import OrderedDict

import torch

from quantize.int_linear_lora import LoRALayer, LoRAQuantLinear
from reassembly.cr_module import CRModule


def get_lws_parameters(sub_layers, round_idx):
    normal_params = []
    normal_params_names = []

    scale_params = []
    scale_params_names = []

    for sub_layer_idx in range(len(sub_layers)):
        for n, p in sub_layers[sub_layer_idx].named_parameters():
            if not p.requires_grad:
                continue
            if "scale" in n:
                scale_params.append(p)
                scale_params_names.append(
                    "round{}_sub{}_{}".format(round_idx, sub_layer_idx, n)
                )
            else:
                normal_params.append(p)
                normal_params_names.append(
                    "round{}_sub{}_{}".format(round_idx, sub_layer_idx, n)
                )
    return normal_params, scale_params, normal_params_names, scale_params_names


def mark_only_lora_as_trainable(
    sub_layers,
    args,
    logger,
    bias="none",
) -> None:
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for n, p in sub_layer.named_parameters():
            p.requires_grad = False

        if args.use_lora:
            for n, p in sub_layer.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
            if bias == "none":
                pass
            elif bias == "all":
                for n, p in sub_layer.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
                    elif "norm" in n:
                        p.requires_grad = True
                    elif "prompt" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in sub_layer.modules():
                    if (
                        isinstance(m, LoRALayer)
                        and hasattr(m, "bias")
                        and m.bias is not None
                    ):
                        m.bias.requires_grad = True
            elif bias == "prompt_only":
                for n, p in sub_layer.named_parameters():
                    if "prompt" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                raise NotImplementedError

        requires_grad_param = []
        for n, p in sub_layer.named_parameters():
            if p.requires_grad == True:
                requires_grad_param.append(n)
        logger.info("Require grad param:")
        logger.info(requires_grad_param)


def obtain_teacher_output(sub_layers, inp, attention_mask, position_ids):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layers[sub_layer_idx].set_quant_state(weight_quant=False, act_quant=False)
        if sub_layer_idx == 0:
            out = sub_layers[sub_layer_idx](
                inp, attention_mask=attention_mask, position_ids=position_ids
            )[0]
        else:
            out = sub_layers[sub_layer_idx](
                out,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
    return out


def obtain_studnet_output(sub_layers, inp, attention_mask, position_ids, args):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layers[sub_layer_idx].set_quant_state(
            weight_quant=args.wbits < 16,
            act_quant=args.abits < 16,
        )
        if sub_layer_idx == 0:
            out = sub_layers[sub_layer_idx](
                inp, attention_mask=attention_mask, position_ids=position_ids
            )[0]
        else:
            out = sub_layers[sub_layer_idx](
                out,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
    return out


def replace_qlayer(config, sub_layers, args, DecoderLayer):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layers[sub_layer_idx] = DecoderLayer(
            config, sub_layers[sub_layer_idx], args
        )
    return sub_layers


def replace_ori_layer(layers, sub_layers, round_idx, args):
    for sub_layer_idx in range(len(sub_layers)):
        layers[round_idx * args.num_layer + sub_layer_idx] = sub_layers[sub_layer_idx]


def to_dev(sub_layers, dev):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(dev)
    return sub_layers


def to_float(sub_layers):
    with torch.no_grad():
        for sub_layer_idx in range(len(sub_layers)):
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].float()
    return sub_layers


def to_half(sub_layers):
    with torch.no_grad():
        for sub_layer_idx in range(len(sub_layers)):
            sub_layers[sub_layer_idx] = sub_layers[sub_layer_idx].to(torch.bfloat16)
    return sub_layers


def load_qlayer_lora_state_dict(sub_layers, state_dict):
    for idx, sub_layer in enumerate(sub_layers):
        sub_layer.load_state_dict(state_dict[idx], strict=False)


def load_qlayer_cr_state_dict(sub_layers, state_dict, dev):
    for idx, sub_layer in enumerate(sub_layers):
        sub_layer.load_state_dict(state_dict[idx], strict=False)
        for name, module in sub_layer.named_modules():
            if isinstance(module, CRModule):
                suffixes = [
                    "outlier_channel_idx",
                    "num_disassembly",
                    "scaling_factors",
                    "src_idx",
                    "dst_idx",
                ]
                for suffix in suffixes:
                    key = f"{name}.{suffix}"
                    value = state_dict[idx][key].to(dev)
                    delattr(module, suffix)
                    module.register_buffer(f"{suffix}", value)


def get_qlayer_lora_state_dict(sub_layers):
    return_dict = OrderedDict()
    for idx, sub_layer in enumerate(sub_layers):
        return_dict[idx] = sub_layer.qllm_lora_state_dict()
    return return_dict


def get_qlayer_cr_state_dict(sub_layers):
    return_dict = OrderedDict()
    for idx, sub_layer in enumerate(sub_layers):
        return_dict[idx] = sub_layer.qllm_sm_state_dict()
    return return_dict


def lora_merge(sub_layers, logger, round_idx, args):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for name, module in sub_layer.named_modules():
            if isinstance(module, (LoRAQuantLinear)):
                logger.info(
                    "Merging weight for layer {}: {}".format(
                        round_idx * args.num_layer + sub_layer_idx, name
                    )
                )
                weight_diff = (
                    module.lora_B.float() @ module.lora_A.float() * module.scaling
                )
                after_training_weight = (module.weight.float() + weight_diff).to(
                    module.weight.dtype
                )
                module.weight.data = after_training_weight.data
                module.merged = True
