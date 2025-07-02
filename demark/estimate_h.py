import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import json
import os
from utils import get_target_word_list, generate_random_choose_probs_on_actual_ctx
import argparse


def estimate_h(
    model,
    model_id,
    tokenizer,
    est_save_dir,
    delta,
    random_word_num,
    max_h=5,
    actual_h=3,
    alpha1=0.2,
    alpha2=10,
    blackbox=False,
):

    word_list = get_target_word_list(tokenizer)

    model.eval()

    res_dict = {"actual_h": actual_h, "same_rate_dict": {}, "delta": delta}
    with torch.no_grad():

        target_tokens = random.sample(word_list, k=random_word_num)
        target_token_ids = [
            tokenizer(i, add_special_tokens=False).input_ids[-1] for i in target_tokens
        ]
        context_token_ids = torch.randint(
            low=1, high=tokenizer.vocab_size, size=(max_h,)
        ).cuda()

        last_final_score = None

        for window_width in range(max_h, 0, -1):
            input_ctx_ids = context_token_ids[-window_width:]
            res_matrix = generate_random_choose_probs_on_actual_ctx(
                model,
                model_id,
                tokenizer,
                ctx_width=actual_h,
                delta=delta,
                ctx_ids=input_ctx_ids,
                token_lists=target_token_ids,
                batch_size=32,
                blackbox=blackbox,
            )
            green_score = ((alpha1 < res_matrix) & (res_matrix < alpha2)).sum(dim=-1)
            red_score = ((-alpha2 < res_matrix) & (res_matrix < -alpha1)).sum(dim=-1)
            final_score = green_score - red_score

            if last_final_score is None:
                last_final_score = final_score
            else:
                same_rate = (
                    (final_score * last_final_score) > 0
                ).sum() / random_word_num
                last_final_score = final_score

                print("window_width:{}, same_rate:{}".format(window_width, same_rate))
                res_dict["same_rate_dict"][window_width] = same_rate.item()

    os.makedirs(est_save_dir, exist_ok=True)
    if blackbox:
        save_filename = os.path.join(
            est_save_dir, model_id.replace("/", "_") + "_blackbox" + ".jsonl"
        )
    else:
        save_filename = os.path.join(
            est_save_dir, model_id.replace("/", "_") + ".jsonl"
        )
    with open(save_filename, "a") as f:
        f.write(json.dumps(res_dict) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual_h", type=int)
    parser.add_argument("--actual_delta", type=int, default=2)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--blackbox", action="store_true")
    parser.add_argument("--random_word_num", type=int, default=50)
    parser.add_argument("--max_h", type=int)
    parser.add_argument("--alpha1", type=float, default=0.2)
    parser.add_argument("--alpha2", type=float, default=10)
    parser.add_argument("--est_save_dir", type=str)

    args = parser.parse_args()

    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()

    estimate_h(
        model,
        model_id,
        tokenizer,
        args.est_save_dir,
        delta=args.actual_delta,
        random_word_num=args.random_word_num,
        max_h=args.max_h,
        actual_h=args.actual_h,
        blackbox=args.blackbox,
    )


if __name__ == "__main__":
    main()
