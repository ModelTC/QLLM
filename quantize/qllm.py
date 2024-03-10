import copy
import gc
import math
import os
import pdb
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import get_scheduler

import utils
from models.int_qllm_llama_layer import QLLMLlamaDecoderLayer
from parallel_utils import map_layers_to_num_gpus
from quantize.learnable_norm import LearnableLlamaRMSNorm
from reassembly.cr_module import CRModule
from train_utils import (
    get_lws_parameters,
    get_qlayer_cr_state_dict,
    get_qlayer_lora_state_dict,
    load_qlayer_cr_state_dict,
    load_qlayer_lora_state_dict,
    lora_merge,
    mark_only_lora_as_trainable,
    obtain_studnet_output,
    obtain_teacher_output,
    replace_ori_layer,
    replace_qlayer,
    to_dev,
    to_float,
    to_half,
)
from utils import oc_feat_dict, oc_maxmin_dict

MB = 1024.0 * 1024.0


def free_hook(handlers):
    for handler in handlers:
        handler.remove()


def free_dict():
    oc_maxmin_dict.clear()
    oc_feat_dict.clear()


def plot_num_additional_channels(args, model):
    layer_names = []
    layer_num_additional_channels = []
    layer_num_channels = []
    for name, module in model.named_modules():
        if isinstance(module, (CRModule)):
            num_additional_channels = module.num_additional_channels
            num_channels = module.num_channels
            layer_names.append(f"{name}")
            layer_num_additional_channels.append(num_additional_channels)
            layer_num_channels.append(num_channels)

    fig = plt.figure()
    plt.plot(
        list(range(len(layer_names))),
        layer_num_additional_channels,
        linestyle="-",
        linewidth=2,
    )
    plt.xlabel("Layer Names")
    plt.ylabel("Number of Additional Channels")
    plt.title("Number of Additional Channels per Layer")
    plt.grid(True)
    plt.xticks(rotation=90)  # Rotate x-axis ticks by 90 degrees
    output_path = os.path.join(args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig("{}/num_additional_channels.pdf".format(output_path))
    plt.tight_layout()
    plt.close()
    print(layer_names)
    print(layer_num_channels)
    print(layer_num_additional_channels)


def register_hook_for_act_min_max(sub_layers):
    handlers = []
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for name, module in sub_layer.named_modules():
            if isinstance(module, (CRModule)):
                print(f"Register act min max hook for {name}")
                module.name = f"layer{sub_layer_idx}_{name}"
                handler = module.register_forward_hook(utils.layer_omax_hook)
                handlers.append(handler)
    return handlers


def register_hook_for_act(sub_layers, args):
    handlers = []
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for name, module in sub_layer.named_modules():
            if isinstance(module, CRModule):
                print(f"Register act hook for {name}")
                module.name = f"layer{sub_layer_idx}_{name}"
                module.bs = args.calibrate_bs
                handler = module.register_forward_hook(utils.layer_o_feature_hook)
                handlers.append(handler)
    return handlers


def perform_reassembly(sub_layers, attention_mask, position_ids):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for name, module in sub_layer.named_modules():
            if isinstance(module, CRModule):
                name = module.name
                x_max, x_min = oc_maxmin_dict[name]
                x_feat = oc_feat_dict[name]

                if "self_attn_layer_norm" in name:
                    sub_layer.self_attn_layer_norm_output_reassembly(
                        x_feat, x_min, x_max, attention_mask, position_ids
                    )
                elif "final_layer_norm" in name:
                    sub_layer.final_layer_norm_output_reassembly(
                        x_feat, x_min, x_max, attention_mask, position_ids
                    )
                elif "input_layernorm" in name:
                    sub_layer.input_layernorm_output_reassembly(
                        x_feat,
                        x_min,
                        x_max,
                        attention_mask,
                        position_ids,
                    )
                elif "post_attention_layernorm" in name:
                    sub_layer.post_attention_layernorm_output_reassembly(
                        x_feat,
                        x_min,
                        x_max,
                        attention_mask,
                        position_ids,
                    )
                elif "down_proj" in name:
                    if x_feat.ndim == 2:
                        x_feat = x_feat.unsqueeze(0)
                    sub_layer.fc1_output_reassembly(
                        x_feat,
                        x_min,
                        x_max,
                        attention_mask,
                        position_ids,
                    )


def perform_reassembly_with_state_dict(sub_layers):
    for sub_layer_idx in range(len(sub_layers)):
        sub_layer = sub_layers[sub_layer_idx]
        for name, module in sub_layer.named_modules():
            if isinstance(module, CRModule):
                name = module.name

                if "input_layernorm" in name:
                    sub_layer.input_layernorm_output_reassembly_with_state_dict()
                elif "post_attention_layernorm" in name:
                    sub_layer.post_attention_layernorm_output_reassembly_with_state_dict()
                elif "down_proj" in name:
                    sub_layer.fc1_output_output_reassembly_with_state_dict()


def plot_act_min_max(args, round_idx, output_name):
    output_path = os.path.join(args.output_dir, f"{output_name}_ac_min_max_plot")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(oc_maxmin_dict, f"{output_path}/round{round_idx}_act_min_max.tar")
    for k, v in oc_maxmin_dict.items():
        fig = plt.figure()
        max_values = v[0].to(torch.float16).detach().cpu().numpy()
        min_values = v[1].to(torch.float16).detach().cpu().numpy()
        x = list(range(len(max_values)))
        plt.plot(x, min_values, label="Minumum")
        plt.plot(x, max_values, label="Maximum")
        plt.xlabel("Channel ID")
        plt.ylabel("Value")
        plt.title(f"Round {round_idx} {k}")
        plt.legend()
        plt.savefig("{}/round{}_{}.pdf".format(output_path, round_idx, k))
        plt.close()


def qllm(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.model or "Llama" in args.model:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QLLMLlamaDecoderLayer
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.model or "Llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)  # take output of fp model as input

    attention_mask = cache["attention_mask"]
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    num_update_steps_per_epoch = math.ceil(args.nsamples / args.batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.resume_reassembly:
        qllm_cr_parameters = torch.load(args.resume_reassembly)
    else:
        qllm_cr_parameters = {}

    if args.resume_lora:
        qllm_lora_parameters = torch.load(args.resume_lora)
    else:
        qllm_lora_parameters = {}

    num_round = model.config.num_hidden_layers // args.num_layer
    for r in range(num_round):
        logger.info(f"=== Round {r} ===")
        logger.info(
            f"=== Start quantize layer{r * args.num_layer}-layer{(r + 1) * args.num_layer - 1} ==="
        )
        sub_layers = layers[r * args.num_layer : (r + 1) * args.num_layer]

        sub_layers = replace_qlayer(lm.model.config, sub_layers, args, DecoderLayer)
        # if args.multigpu:
        #     map_layers_to_num_gpus(sub_layers, args.num_gpu)
        # else:
        sub_layers = to_dev(sub_layers, dev)

        logger.info("Student layer")
        logger.info(sub_layers)

        act_min_max_handlers = register_hook_for_act_min_max(sub_layers)
        act_handlers = register_hook_for_act(sub_layers, args)

        # obtain output of full-precision model
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for j in range(args.nsamples):
                    fp_inps[j] = obtain_teacher_output(
                        sub_layers,
                        fp_inps[j].unsqueeze(0),
                        attention_mask,
                        position_ids,
                    )[0]

        free_hook(act_min_max_handlers + act_handlers)
        if args.resume_reassembly and r in qllm_cr_parameters:
            sub_layers_dict = qllm_cr_parameters[r]
            load_qlayer_cr_state_dict(sub_layers, sub_layers_dict, dev)
            with torch.no_grad():
                perform_reassembly_with_state_dict(sub_layers)
        else:
            with torch.no_grad():
                perform_reassembly(sub_layers, attention_mask, position_ids)
            sub_layers_dict = get_qlayer_cr_state_dict(sub_layers)
            qllm_cr_parameters[r] = sub_layers_dict
            torch.save(
                qllm_cr_parameters,
                os.path.join(args.output_dir, f"qllm_cr_parameters.pth"),
            )
        if args.plot_act_max:
            plot_act_min_max(args, r, "before_learning")
        free_dict()

        sub_layers = to_float(sub_layers)

        # Optimizer
        if args.use_lora:
            mark_only_lora_as_trainable(sub_layers, args, logger, "all")

        (
            normal_params,
            scale_params,
            normal_params_names,
            scale_params_names,
        ) = get_lws_parameters(sub_layers, r)
        optimizer = torch.optim.AdamW(
            [
                {"params": normal_params, "lr": args.lr},
                {"params": scale_params, "lr": args.lr},
            ],
            weight_decay=args.wd,
        )
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=args.max_train_steps,
        )
        logger.info(optimizer)
        logger.info("Normal parameter")
        logger.info(normal_params_names)
        logger.info("Step size parameter")
        logger.info(scale_params_names)

        updated_steps = 0
        losses = []

        if args.resume_lora and r in qllm_lora_parameters:
            sub_layers_dict = qllm_lora_parameters[r]
            load_qlayer_lora_state_dict(sub_layers, sub_layers_dict)
        else:
            for e in range(args.epochs):
                start_time = time.time()
                randperm = torch.randperm(args.nsamples)
                num_batch = args.nsamples // args.batch_size

                epoch_losses = []

                for batch_idx in range(num_batch):
                    sample_idxes = randperm[
                        batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
                    ]

                    batch_loss = 0.0
                    optimizer.zero_grad()
                    for sample_id in sample_idxes:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            inp = quant_inps[sample_id].unsqueeze(0)

                            # student
                            out = obtain_studnet_output(
                                sub_layers, inp, attention_mask, position_ids, args
                            )

                            loss = loss_func(
                                out, fp_inps[sample_id].to(out.device)
                            ) / len(sample_idxes)

                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")
                            pdb.set_trace()

                        loss.backward()
                        batch_loss += loss.item()

                    optimizer.step()
                    lr_scheduler.step()

                    losses.append(batch_loss)
                    epoch_losses.append(batch_loss)

                    current_memory = torch.cuda.memory_allocated() / MB
                    max_memory = torch.cuda.max_memory_allocated() / MB

                    updated_steps += 1
                    if updated_steps >= args.max_train_steps:
                        # break when the number of steps is reached, typically in hundreds
                        break

                epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(
                    f"Round {r} epoch {e} loss:{epoch_mean_loss} max memory_allocated: {max_memory}MB current memory_allocated: {current_memory}MB time: {time.time() - start_time}s"
                )
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()

            fig = plt.figure()
            plt.plot(losses)
            plt.show()
            plt.savefig(
                "{}/layer{}-layer{}-loss.pdf".format(
                    args.output_dir, r * args.num_layer, (r + 1) * args.num_layer - 1
                )
            )
            plt.close()

            logger.info("Current GPU memory {}MB".format(current_memory))
            logger.info("Max GPU memory {}MB".format(max_memory))

            sub_layers_dict = get_qlayer_lora_state_dict(sub_layers)
            qllm_lora_parameters[r] = sub_layers_dict
            torch.save(
                qllm_lora_parameters,
                os.path.join(args.output_dir, f"qllm_lora_parameters.pth"),
            )

        # plot act min max after learning
        if args.plot_act_max:
            handlers = register_hook_for_act_min_max(sub_layers)

        if args.use_lora:
            with torch.no_grad():
                lora_merge(sub_layers, logger, r, args)

        # get next layer input
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for sample_idx in range(args.nsamples):
                    quant_inps[sample_idx] = obtain_studnet_output(
                        sub_layers,
                        quant_inps[sample_idx].unsqueeze(0),
                        attention_mask,
                        position_ids,
                        args,
                    )

        if args.plot_act_max:
            plot_act_min_max(args, r, "after_learning")
            free_hook(handlers)
            free_dict()

        sub_layers = to_half(sub_layers)
        sub_layers = to_dev(sub_layers, "cpu")
        replace_ori_layer(layers, sub_layers, r, args)

        gc.collect()
        torch.cuda.empty_cache()

    if args.plot_num_additional_channels:
        plot_num_additional_channels(args, model)

    tune_last_ln(lm, quant_inps, fp_inps, logger, args)

    del quant_inps
    del fp_inps
    model.config.use_cache = use_cache
    gc.collect()
    torch.cuda.empty_cache()
    return model


def qllm_without_train(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.model or "Llama" in args.model:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QLLMLlamaDecoderLayer
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")

    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.model or "Llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)  # take output of fp model as input

    attention_mask = cache["attention_mask"]
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    num_update_steps_per_epoch = math.ceil(args.nsamples / args.batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_losses = []

    num_round = model.config.num_hidden_layers // args.num_layer
    for r in range(num_round):
        logger.info(f"=== Round {r} ===")
        logger.info(
            f"=== Start quantize layer{r * args.num_layer}-layer{(r + 1) * args.num_layer - 1} ==="
        )
        sub_layers = layers[r * args.num_layer : (r + 1) * args.num_layer]

        sub_layers = to_dev(sub_layers, dev)
        sub_layers = replace_qlayer(lm.model.config, sub_layers, args, DecoderLayer)

        logger.info("Student layer")
        logger.info(sub_layers)

        act_min_max_handlers = register_hook_for_act_min_max(sub_layers)
        act_handlers = register_hook_for_act(sub_layers, args)

        # obtain output of full-precision model
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for j in range(args.nsamples):
                    fp_inps[j] = obtain_teacher_output(
                        sub_layers,
                        fp_inps[j].unsqueeze(0),
                        attention_mask,
                        position_ids,
                    )[0]

        free_hook(act_min_max_handlers + act_handlers)
        with torch.no_grad():
            perform_reassembly(sub_layers, attention_mask, position_ids)
        if args.plot_act_max:
            plot_act_min_max(args, r, "before_learning")
        free_dict()

        # plot act min max after learning
        if args.plot_act_max:
            handlers = register_hook_for_act_min_max(sub_layers)

        # get next layer input
        total_loss = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for sample_idx in range(args.nsamples):
                    quant_inps[sample_idx] = obtain_studnet_output(
                        sub_layers,
                        quant_inps[sample_idx].unsqueeze(0),
                        attention_mask,
                        position_ids,
                        args,
                    )
                    loss = loss_func(fp_inps[j], quant_inps[j])
                    total_loss += loss.item()
                total_loss = total_loss / args.nsamples

        print("Current reconstruction error: {}".format(total_loss))
        total_losses.append(total_loss)

        if args.plot_act_max:
            plot_act_min_max(args, r, "after_learning")
            free_hook(handlers)
            free_dict()

        sub_layers = to_dev(sub_layers, "cpu")
        replace_ori_layer(layers, sub_layers, r, args)

        # del out
        gc.collect()
        torch.cuda.empty_cache()

    if args.plot_num_additional_channels:
        plot_num_additional_channels(args, model)

    print("Total losses:")
    print(total_losses)

    del quant_inps
    del fp_inps
    model.config.use_cache = use_cache
    gc.collect()
    torch.cuda.empty_cache()
    return model


def tune_last_ln(lm, quant_inps, fp_inps, logger, args):
    model = lm.model
    dev = lm.device

    logger.info(f"=== Tuning the last ln ===")
    model.model.norm = model.model.norm.to(dev)
    if args.learn_ln_no_bias:
        last_ln = model.model.norm
    else:
        last_ln = LearnableLlamaRMSNorm(model.model.norm)

    logger.info(last_ln)
    # obtain output of full-precision model
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for j in range(args.nsamples):
                fp_inps[j] = last_ln(fp_inps[j].unsqueeze(0))

    last_ln = last_ln.float()
    params = []
    params_names = []
    for n, p in last_ln.named_parameters():
        params.append(p)
        params_names.append(n)

    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": args.lr}], weight_decay=args.wd
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )
    logger.info(optimizer)
    logger.info(params_names)

    updated_steps = 0
    losses = []
    loss_func = torch.nn.MSELoss()

    for e in range(args.epochs):
        start_time = time.time()
        randperm = torch.randperm(args.nsamples)
        num_batch = args.nsamples // args.batch_size

        epoch_losses = []

        for batch_idx in range(num_batch):
            sample_idxes = randperm[
                batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
            ]

            batch_loss = 0.0
            optimizer.zero_grad()
            for sample_id in sample_idxes:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    inp = quant_inps[sample_id].unsqueeze(0)

                    # student
                    out = last_ln(inp)
                    loss = loss_func(out, fp_inps[sample_id]) / len(sample_idxes)

                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()

                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()

            losses.append(batch_loss)
            epoch_losses.append(batch_loss)

            current_memory = torch.cuda.memory_allocated() / MB
            max_memory = torch.cuda.max_memory_allocated() / MB

            updated_steps += 1
            if updated_steps >= args.max_train_steps:
                # break when the number of steps is reached, typically in hundreds
                break

        epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(
            f"Last LN epoch {e} loss:{epoch_mean_loss} max memory_allocated: {max_memory}MB current memory_allocated: {current_memory}MB time: {time.time() - start_time}s"
        )

    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    fig = plt.figure()
    plt.plot(losses)
    plt.show()
    plt.savefig("{}/last_ln-loss.pdf".format(args.output_dir))
    plt.close()

    logger.info("Current GPU memory {}MB".format(current_memory))
    logger.info("Max GPU memory {}MB".format(max_memory))

    last_ln = last_ln.to(torch.bfloat16)
    last_ln = last_ln.to("cpu")
    model.model.norm = last_ln

    del out
    gc.collect()
    torch.cuda.empty_cache()
