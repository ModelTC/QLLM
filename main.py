import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import SchedulerType

import utils
from datautils import get_loaders
from eval import evaluate
from models.LMClass import LMClass
from quantize.qllm import qllm, qllm_without_train

torch.backends.cudnn.benchmark = True

net_choices = [
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument(
        "--cache_dir",
        default="./cache",
        type=str,
        help="cache dir of dataset, leading to faster debug",
    )
    parser.add_argument(
        "--output_dir", default="../log/", type=str, help="direction of logging file"
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="direction for saving fake quantization model",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument(
        "--seed", type=int, default=2, help="Seed for sampling the calibration data."
    )
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--symmetric", default=False, action="store_true", help="symmetric quantization"
    )
    parser.add_argument(
        "--a_dynamic_method", type=str, default="per_token", choices=["per_token"]
    )
    parser.add_argument(
        "--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--multigpu", action="store_true", help="at eval, map model to multiple gpus"
    )
    parser.add_argument("--num_gpu", type=int, default=1)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--plot_act_max",
        action="store_true",
        help="whether to plot scale and shift.",
    )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="do not train model weights or lora weight.",
    )
    parser.add_argument("--channel_ratio", type=float, default=0.1)
    parser.add_argument(
        "--plot_num_additional_channels",
        action="store_true",
        help="whether to plot scale and shift.",
    )
    parser.add_argument("--resume_reassembly", type=str, default=None)
    parser.add_argument("--resume_lora", type=str, default=None)
    parser.add_argument("--calibrate_bs", type=int, default=4, help="batch size.")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument(
        "--learn_ln_no_bias",
        action="store_true",
        help="whether to use shift.",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="whether to use fp 16",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)

    # load model
    args.net = args.model.split("/")[-1]
    assert args.net in net_choices
    args.model_family = args.net.split("-")[0]
    lm = LMClass(args, logger)
    lm.seqlen = args.seq_len
    lm.model.eval()
    logger.info("=== start quantization ===")
    tick = time.time()

    # load calibration dataset
    cache_dataloader = f"{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}_{args.seq_len}.cache"
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logger.info(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = 0
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    if args.no_train:
        qllm_without_train(
            lm,
            args,
            dataloader,
            logger,
        )
    else:
        qllm(lm, args, dataloader, logger)

    logger.info(time.time() - tick)

    if args.save_dir:
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)

    evaluate(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
