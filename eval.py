import json
import os
import time
from pprint import pprint

import numpy as np
import ray
import shortuuid
import torch
import torch.nn as nn
from fastchat.model import get_conversation_template
from tqdm import tqdm

from categories import categories, subcategories
from datautils import get_loaders
from lm_eval import evaluator
from parallel_utils import map_layers_to_multi_gpus


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.model:
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif (
            "llama" in args.model
            or "Llama" in args.model
            or "vicuna" in args.model
            or "alpaca" in args.model
        ):
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.model:
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif (
            "llama" in args.model
            or "Llama" in args.model
            or "vicuna" in args.model
            or "alpaca" in args.model
        ):
            lm.model = lm.model.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "ptb", "c4"]:
            cache_testloader = (
                f"{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache"
            )
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                    lm.device
                )
                if "opt" in args.model:
                    outputs = lm.model.model.decoder(batch)
                elif (
                    "llama" in args.model
                    or "Llama" in args.model
                    or "vicuna" in args.model
                    or "alpaca" in args.model
                ):
                    outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f"{dataset} : {ppl.item()}")
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if "hendrycksTest" in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {
                subcat: []
                for subcat_lists in subcategories.values()
                for subcat in subcat_lists
            }
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results["results"].keys():
                if not "hendrycksTest" in key:
                    continue
                subject = key.split("-")[-1]
                cors = t_results["results"][key]["acc"]
                cors_norm = t_results["results"][key]["acc_norm"]
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)

            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))
    return results


@torch.no_grad()
def test_throughput(lm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.model:
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.model or "Llama" in args.model:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.model:
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.model or "Llama" in args.model:
            lm.model = lm.model.to(lm.device)

    dataset = "wikitext2"
    cache_testloader = (
        f"{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache"
    )
    if os.path.exists(cache_testloader):
        testloader = torch.load(cache_testloader)
        logger.info(f"load calibration from {cache_testloader}")
    else:
        dataloader, testloader = get_loaders(
            dataset,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(testloader, cache_testloader)
    if "c4" in dataset:
        testenc = testloader
    else:
        testenc = testloader.input_ids

    nsamples = testenc.numel() // lm.seqlen
    use_cache = lm.model.config.use_cache
    lm.model.config.use_cache = False
    lm.model.eval()
    nlls = []

    # warmup
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
        if "opt" in args.model:
            outputs = lm.model.model.decoder(batch)
        elif "llama" in args.model or "Llama" in args.model:
            outputs = lm.model.model(batch)
        hidden_states = outputs[0]
        logits = lm.model.lm_head(hidden_states)

    start_time = time.time()
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
        if "opt" in args.model:
            outputs = lm.model.model.decoder(batch)
        elif "llama" in args.model or "Llama" in args.model:
            outputs = lm.model.model(batch)
        hidden_states = outputs[0]
        logits = lm.model.lm_head(hidden_states)
    end_time = time.time()
    avg_time = (end_time - start_time) / nsamples
    print(f"Avg time for seq_len {lm.seqlen} is {avg_time}s")


def chat_run_eval(lm, model_name, question_file, answer_file, num_gpus=1):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    ans_jsons = get_model_answers(lm, model_name, ques_jsons)

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


@torch.inference_mode()
def get_model_answers(lm, model_name, question_jsons):
    lm.model = lm.model.to(lm._device)

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        conv = get_conversation_template(model_name)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = lm.tokenizer([prompt]).input_ids
        output_ids = lm.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = lm.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {},
            }
        )
    return ans_jsons
