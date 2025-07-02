import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import tqdm

from utils import (
    get_ngram_red_green_list,
    detect_wm_dipmark,
    detect_wm_KGW,
    get_target_word_list,
    get_top_k_indices,
    dipmark_logit_processor,
    cal_results,
)
from predict_red_green_list import (
    detect_red_green_list_KGW,
    detect_red_green_list_dipmark,
)
from generation_dataset import GenerationPrompts
import json
import argparse
import os
import math


def generate_with_watermark_removal_KGW(
    model,
    model_id,
    tokenizer,
    raw_prompt,
    prompt_ids,
    max_new_token=100,
    total_word_num=0,
    ctx_width=3,
    delta_actual=2,
    delta_predicted=2,
    sample_strategy=None,
    print_all=False,
    delta_reduce_multiplier=1,
):

    generated_token_ids = []
    total_statistics, possible_token_statistics = (
        torch.zeros((4,)).cuda(),
        torch.zeros((4,)).cuda(),
    )

    for gen_idx in tqdm.tqdm(range(max_new_token)):
        if print_all:
            print("running step {}".format(gen_idx + 1), flush=True)
        new_prompt_ids = torch.cat(
            [
                prompt_ids,
                torch.tensor(generated_token_ids, dtype=torch.long).cuda().view(1, -1),
            ],
            dim=1,
        )
        forward_res = model(new_prompt_ids)
        last_token_logits = forward_res.logits[0, -1, :]

        cur_green_list, cur_red_list = get_ngram_red_green_list(
            new_prompt_ids[0, -ctx_width:],
            last_token_logits.shape[-1],
            new_prompt_ids.device,
        )

        raw_logits = last_token_logits.clone().detach()

        last_token_logits[cur_green_list] += delta_actual

        if delta_predicted > 0:
            new_green_list, new_total_statistics, new_possible_token_statistics = (
                detect_red_green_list_KGW(
                    model_id,
                    new_prompt_ids,
                    ctx_width,
                    total_word_num=total_word_num,
                    model=model,
                    tokenizer=tokenizer,
                    delta_actual=delta_actual,
                    delta_predicted=delta_predicted,
                    sample_strategy=sample_strategy,
                    print_all=print_all,
                    vocab_size=last_token_logits.shape[-1],
                )
            )

            total_statistics += new_total_statistics
            possible_token_statistics += new_possible_token_statistics

        if sample_strategy == "top_k":
            to_sample_tokens = get_top_k_indices(last_token_logits, k=total_word_num)
        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")

        if delta_predicted > 0:
            last_token_logits[new_green_list] -= (
                delta_predicted * delta_reduce_multiplier
            )

        mask = torch.zeros_like(last_token_logits).cuda()
        mask[to_sample_tokens] = 1
        new_probs = F.softmax(last_token_logits, dim=-1)

        new_probs = new_probs * mask
        new_probs = new_probs / new_probs.sum()

        raw_probs = F.softmax(raw_logits, dim=-1)
        raw_probs = raw_probs * mask
        raw_probs = raw_probs / raw_probs.sum()

        new_token = torch.multinomial(new_probs, 1)
        generated_token_ids.append(new_token)
        if new_token == tokenizer.eos_token_id:
            break

    if print_all:
        print("generation results:")
        print(
            tokenizer.decode(
                torch.cat(
                    [prompt_ids, torch.tensor(generated_token_ids).cuda().view(1, -1)],
                    dim=1,
                )[0]
            )
        )
        print("Final all token results:")
        print(cal_results(total_statistics))
        print("Final posible token results:")
        print(cal_results(possible_token_statistics), flush=True)

    detect_wm_result = detect_wm_KGW(
        ctx_width=ctx_width,
        prompt_ids=prompt_ids,
        generated_token_ids=generated_token_ids,
        tokenizer=tokenizer,
    )

    result_dict = {}
    result_dict["prompt"] = tokenizer.decode(prompt_ids[0])
    result_dict["raw_prompt"] = raw_prompt
    # print(generated_token_ids.shape)
    result_dict["generated_text"] = tokenizer.decode(
        torch.tensor(generated_token_ids).long()
    )
    result_dict["detection_result"] = detect_wm_result

    result_dict["final_all_token_results"] = cal_results(total_statistics)
    result_dict["final_possible_token_results"] = cal_results(possible_token_statistics)
    result_dict["total_statistics"] = total_statistics.cpu().tolist()
    result_dict["possible_token_statistics"] = possible_token_statistics.cpu().tolist()

    return result_dict


def generate_with_watermark_removal_dipmark(
    model,
    model_id,
    tokenizer,
    raw_prompt,
    prompt_ids,
    max_new_token=100,
    total_word_num=0,
    ctx_width=3,
    delta_predicted=2,
    sample_strategy=None,
    print_all=False,
    delta_reduce_multiplier=1,
    dipmark_alpha=0.3,
):
    generated_token_ids = []
    # total_acc,total_cnt,possible_token_acc,possible_token_cnt=0,0,0,0
    total_statistics, possible_token_statistics = (
        torch.zeros((4,)).cuda(),
        torch.zeros((4,)).cuda(),
    )

    for gen_idx in tqdm.tqdm(range(max_new_token)):
        if print_all:
            print("running step {}".format(gen_idx + 1), flush=True)
        new_prompt_ids = torch.cat(
            [
                prompt_ids,
                torch.tensor(generated_token_ids, dtype=torch.long).cuda().view(1, -1),
            ],
            dim=1,
        )

        forward_res = model(new_prompt_ids)
        last_token_logits = forward_res.logits[0, -1, :]

        cur_green_list, cur_red_list = get_ngram_red_green_list(
            new_prompt_ids[0, -ctx_width:],
            last_token_logits.shape[-1],
            new_prompt_ids.device,
        )

        # last_token_logits[cur_green_list]+=delta_actual
        last_token_logits = dipmark_logit_processor(
            last_token_logits, cur_green_list, cur_red_list, dipmark_alpha
        )

        last_token_logits[tokenizer.eos_token_id] = (
            -1e5
        )  # discourage eos #remove this for other tasks

        if delta_predicted > 0:
            new_green_list, new_total_statistics, new_possible_token_statistics = (
                detect_red_green_list_dipmark(
                    model_id,
                    new_prompt_ids,
                    ctx_width,
                    total_word_num=total_word_num,
                    model=model,
                    tokenizer=tokenizer,
                    delta_predicted=delta_predicted,
                    sample_strategy=sample_strategy,
                    print_all=print_all,
                    vocab_size=last_token_logits.shape[-1],
                    dipmark_alpha=dipmark_alpha,
                )
            )

            total_statistics += new_total_statistics
            possible_token_statistics += new_possible_token_statistics

        if sample_strategy == "top_k":
            to_sample_tokens = get_top_k_indices(last_token_logits, k=total_word_num)
        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")

        if delta_predicted > 0:
            last_token_logits[new_green_list] -= (
                delta_predicted * delta_reduce_multiplier
            )

        mask = torch.zeros_like(last_token_logits).cuda()
        mask[to_sample_tokens] = 1
        new_probs = F.softmax(last_token_logits, dim=-1)
        new_probs = new_probs * mask
        new_probs = new_probs / new_probs.sum()
        new_token = torch.multinomial(new_probs, 1)
        generated_token_ids.append(new_token)
        if new_token == tokenizer.eos_token_id:
            break

    if print_all:
        print("generation results:")
        print(
            tokenizer.decode(
                torch.cat(
                    [prompt_ids, torch.tensor(generated_token_ids).cuda().view(1, -1)],
                    dim=1,
                )[0]
            )
        )
        print("Final all token results:")
        print(cal_results(total_statistics))
        print("Final posible token results:")
        print(cal_results(possible_token_statistics), flush=True)

    detect_wm_result = detect_wm_dipmark(
        ctx_width=ctx_width,
        prompt_ids=prompt_ids,
        generated_token_ids=generated_token_ids,
        tokenizer=tokenizer,
    )

    result_dict = {}
    result_dict["prompt"] = tokenizer.decode(prompt_ids[0])
    result_dict["raw_prompt"] = raw_prompt
    result_dict["generated_text"] = tokenizer.decode(
        torch.tensor(generated_token_ids).long()
    )
    result_dict["detection_result"] = detect_wm_result

    result_dict["final_all_token_results"] = cal_results(total_statistics)
    result_dict["final_possible_token_results"] = cal_results(possible_token_statistics)
    result_dict["total_statistics"] = total_statistics.cpu().tolist()
    result_dict["possible_token_statistics"] = possible_token_statistics.cpu().tolist()

    return result_dict


def main(args):

    model_id = args.model_id
    prompt_set_name = args.prompt_set_name

    cur_save_dir = os.path.join(args.save_dir, prompt_set_name)

    if args.delta_predicted > 0:
        assert args.delta_actual > 0

    save_exp_dir = os.path.join(
        cur_save_dir, model_id.replace("/", "_"), args.save_name_suffix
    )

    separate_save_dir = os.path.join(save_exp_dir, "separate_results")
    os.makedirs(separate_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
    model.eval()

    gen_dataset = GenerationPrompts(
        model_id=model_id, prompt_set_name=prompt_set_name, tokenizer=tokenizer
    )

    num_per_task = math.ceil(len(gen_dataset) / args.total_task_num)
    args.start_idx = num_per_task * args.task_idx
    args.end_idx = num_per_task * (args.task_idx + 1)
    args.end_idx = min(args.end_idx, len(gen_dataset))

    generation_args = {
        "delta_actual": args.delta_actual,
        "delta_predicted": args.delta_predicted,
        "ctx_width": args.ctx_width,
        "max_new_token": args.max_new_token,
        "total_word_num": args.total_word_num,
        "sample_strategy": args.sample_strategy,
        "delta_reduce_multiplier": args.delta_reduce_multiplier,
    }

    with torch.no_grad():
        save_filename = os.path.join(
            separate_save_dir, str(args.start_idx) + "_" + str(args.end_idx) + ".jsonl"
        )
        # skip_num=0
        generated_idx_list = []
        if os.path.exists(save_filename):
            with open(save_filename, "r") as f:
                # skip_num+=len(f.readlines())
                lines = f.readlines()
            for line in lines:
                cur_res = json.loads(line)
                generated_idx_list.append(cur_res["prompt_idx"])

        for idx in range(args.start_idx, args.end_idx):
            if idx in generated_idx_list:
                continue

            prompt_ids, raw_prompt, prompt_idx = gen_dataset[idx]
            print("running idx: ", prompt_idx)
            if args.wm_type == "KGW":
                result_dict = generate_with_watermark_removal_KGW(
                    model,
                    model_id,
                    tokenizer,
                    raw_prompt,
                    prompt_ids,
                    max_new_token=args.max_new_token,
                    total_word_num=args.total_word_num,
                    ctx_width=args.ctx_width,
                    delta_actual=args.delta_actual,
                    delta_predicted=args.delta_predicted,
                    sample_strategy=args.sample_strategy,
                    delta_reduce_multiplier=args.delta_reduce_multiplier,
                )
            elif args.wm_type == "dipmark":
                result_dict = generate_with_watermark_removal_dipmark(
                    model,
                    model_id,
                    tokenizer,
                    raw_prompt,
                    prompt_ids,
                    max_new_token=args.max_new_token,
                    total_word_num=args.total_word_num,
                    ctx_width=args.ctx_width,
                    delta_predicted=args.delta_predicted,
                    sample_strategy=args.sample_strategy,
                    delta_reduce_multiplier=args.delta_reduce_multiplier,
                    dipmark_alpha=args.dipmark_alpha,
                )
            else:
                raise ValueError(f"Unknown wm_type: {args.wm_type}")

            result_dict["generation_args"] = generation_args
            result_dict["prompt_idx"] = prompt_idx

            with open(
                os.path.join(
                    separate_save_dir,
                    str(args.start_idx) + "_" + str(args.end_idx) + ".jsonl",
                ),
                "a",
            ) as f:
                f.write(json.dumps(result_dict) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_idx", type=int)
    parser.add_argument("--total_task_num", type=int)
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--prompt_set_name", type=str, default="mmw_book_report")

    parser.add_argument("--max_new_token", type=int, default=100)
    parser.add_argument("--total_word_num", type=int, default=20)
    parser.add_argument("--ctx_width", type=int, default=3)
    parser.add_argument("--delta_actual", type=float, default=2)
    parser.add_argument("--delta_predicted", type=float, default=0)
    parser.add_argument("--delta_reduce_multiplier", type=float, default=1.0)

    parser.add_argument("--dipmark_alpha", type=float, default=0.3)

    parser.add_argument("--wm_type", type=str, choices=["dipmark", "KGW"])

    parser.add_argument("--sample_strategy", type=str, default="top_k")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_name_suffix", type=str)

    args = parser.parse_args()

    main(args)
