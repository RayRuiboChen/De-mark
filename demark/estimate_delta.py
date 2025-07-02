import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

from utils import (
    get_target_word_list,
    generate_random_choose_probs,
    get_ngram_red_green_list,
)
import tqdm
import argparse


def estimate_delta(
    model,
    model_id,
    tokenizer,
    actual_delta,
    random_word_num,
    repeat_time,
    window_width,
    alpha1,
    alpha2,
    gamma,
    blackbox,
):

    word_list = get_target_word_list(tokenizer, model_id=model_id)
    model.eval()
    with torch.no_grad():
        total_delta = 0
        total_cnt = 0
        for repeat_idx in range(repeat_time):
            context_token_ids = torch.randint(
                low=1, high=tokenizer.vocab_size, size=(window_width,)
            ).cuda()

            target_tokens = random.sample(word_list, k=random_word_num)
            target_token_ids = [
                tokenizer(i, add_special_tokens=False).input_ids[-1]
                for i in target_tokens
            ]

            cur_green_list, _ = get_ngram_red_green_list(
                context_tokens=context_token_ids,
                vocab_size=tokenizer.vocab_size,
                device=context_token_ids.device,
            )
            res_matrix = generate_random_choose_probs(
                model,
                model_id,
                tokenizer,
                cur_green_list=cur_green_list,
                delta=actual_delta,
                ctx_ids=context_token_ids,
                token_lists=target_token_ids,
                batch_size=32,
                blackbox=blackbox,
            )

            green_score = ((alpha1 < res_matrix) & (res_matrix < alpha2)).sum(dim=-1)
            red_score = ((-alpha2 < res_matrix) & (res_matrix < -alpha1)).sum(dim=-1)
            final_score = green_score - red_score

            for idx1 in range(random_word_num):
                if final_score[idx1] > gamma * random_word_num:
                    for idx2 in range(random_word_num):
                        if final_score[idx2] < -gamma * random_word_num:
                            total_cnt += 1
                            total_delta += res_matrix[idx1][idx2]
    if total_cnt != 0:
        return (total_delta / total_cnt).item()
    else:  # numerical stability
        return estimate_delta(
            model, model_id, tokenizer, random_word_num, repeat_time, window_width
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", type=int)
    parser.add_argument("--repeat_time", type=int)
    parser.add_argument("--random_word_num", type=int)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--actual_delta", type=float, default=2)
    parser.add_argument("--context_width_h", type=int, default=3)
    parser.add_argument("--alpha1", type=float, default=0.2)
    parser.add_argument("--alpha2", type=float, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--blackbox", action="store_true")

    args = parser.parse_args()
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()

    delta_list = []
    print("running estimation...")
    for test_idx in tqdm.tqdm(range(args.run_num)):
        cur_delta = estimate_delta(
            model,
            model_id,
            tokenizer,
            actual_delta=args.actual_delta,
            random_word_num=args.random_word_num,
            repeat_time=args.repeat_time,
            window_width=args.context_width_h,
            alpha1=args.alpha1,
            alpha2=args.alpha2,
            gamma=args.gamma,
            blackbox=args.blackbox,
        )
        delta_list.append(cur_delta)

    print("-" * 80)
    print(
        f"Actual delta: {args.actual_delta}\nEstimated delta: {sum(delta_list)/args.run_num}"
    )


if __name__ == "__main__":
    main()
