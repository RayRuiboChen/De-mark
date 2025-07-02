import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from utils import (
    get_ngram_red_green_list,
    get_top_k_indices,
    generate_random_choose_probs,
    generate_random_choose_probs_dipmark,
    cal_results,
    dipmark_logit_processor,
)


def detect_red_green_list_KGW(
    model_id,
    prompt_ids,
    ctx_width,
    total_word_num,
    model,
    tokenizer,
    delta_actual=2,
    delta_predicted=2,
    sample_strategy=None,
    print_all=False,
    predefined_possible_token_ids=None,
    blackbox=False,
    vocab_size=-1,
):
    assert prompt_ids.shape[0] == 1
    assert ctx_width <= prompt_ids.shape[1]
    model.eval()

    with torch.no_grad():

        ctx_ids = prompt_ids[0, -ctx_width:]

        forward_res = model(prompt_ids)
        last_token_logits = forward_res.logits[0, -1, :]

        cur_green_list, cur_red_list = get_ngram_red_green_list(
            ctx_ids, vocab_size, last_token_logits.device
        )
        last_token_logits[cur_green_list] += delta_actual

        if sample_strategy == "top_k":
            assert predefined_possible_token_ids is None
            possible_token_ids = get_top_k_indices(last_token_logits, k=total_word_num)
            token_lists = list(set(possible_token_ids.cpu().tolist()))

        elif sample_strategy == "predefined_token_list":
            possible_token_ids = predefined_possible_token_ids
            token_lists = list(set(possible_token_ids.cpu().tolist()))
        else:
            print("Unknown sample strategy: ", sample_strategy)
            raise NotImplementedError

        res_matrix_raw = generate_random_choose_probs(
            model,
            model_id,
            tokenizer,
            cur_green_list,
            delta_actual,
            ctx_ids,
            token_lists,
            batch_size=32,
            blackbox=blackbox,
            sample_num=10,
        )
        res_matrix = torch.zeros_like(res_matrix_raw).cuda()

        for idx1 in range(len(token_lists)):
            for idx2 in range(idx1 + 1, len(token_lists)):
                s1 = res_matrix_raw[idx1][idx2]
                s1_abs = torch.abs(s1)
                pairwise_score = torch.sign(s1) * torch.minimum(
                    s1_abs / delta_predicted, delta_predicted / s1_abs
                )

                res_matrix[idx1][idx2] = pairwise_score
                res_matrix[idx2][idx1] = -pairwise_score

        total_statistics = torch.zeros((4,)).cuda()  # [tp,tn,fp,fn]
        possible_token_statistics = torch.zeros((4,)).cuda()  # [tp,tn,fp,fn]

        res_matrix_sum = res_matrix.sum(dim=-1)

        token_scores = torch.zeros((len(token_lists),)).cuda()

        for idx1 in range(len(token_lists)):
            for idx2 in range(len(token_lists)):
                token_scores[idx1] += (
                    torch.abs(res_matrix_sum[idx2]) * res_matrix[idx1][idx2]
                )
        final_green_list_tokens = []  # high precision, relatively low recall
        for idx in range(len(token_lists)):
            if token_scores[idx] > 0:
                final_green_list_tokens.append(idx)

        for idx in range(len(token_lists)):
            if idx in final_green_list_tokens:  # predicted as green
                if token_lists[idx] in cur_green_list:  # True positive
                    total_statistics[0] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[0] += 1
                else:  # False positive
                    total_statistics[2] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[2] += 1

            else:  # predicted as red
                if token_lists[idx] in cur_red_list:  # True negative
                    total_statistics[1] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[1] += 1
                else:  # False negative
                    total_statistics[3] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[3] += 1

        if print_all:
            print("Local all token results:")
            print(cal_results(total_statistics))
            print("Local posible token results:")
            print(cal_results(possible_token_statistics), flush=True)

        detected_green_token_ids = []
        for idx in range(len(token_lists)):
            if (token_lists[idx] in possible_token_ids) and token_scores[idx] > 0:
                detected_green_token_ids.append(token_lists[idx])

        return detected_green_token_ids, total_statistics, possible_token_statistics


def detect_red_green_list_dipmark(
    model_id,
    prompt_ids,
    ctx_width,
    total_word_num,
    model,
    tokenizer,
    delta_predicted=2,
    sample_strategy=None,
    print_all=False,
    predefined_possible_token_ids=None,
    blackbox=False,
    vocab_size=-1,
    dipmark_alpha=0.5,
):
    assert prompt_ids.shape[0] == 1
    assert ctx_width <= prompt_ids.shape[1]
    model.eval()

    with torch.no_grad():

        ctx_ids = prompt_ids[0, -ctx_width:]

        forward_res = model(prompt_ids)
        last_token_logits = forward_res.logits[0, -1, :]
        cur_green_list, cur_red_list = get_ngram_red_green_list(
            ctx_ids, vocab_size, last_token_logits.device
        )

        last_token_logits = dipmark_logit_processor(
            last_token_logits, cur_green_list, cur_red_list, alpha=dipmark_alpha
        )
        if sample_strategy == "top_k":
            assert predefined_possible_token_ids is None
            possible_token_ids = get_top_k_indices(last_token_logits, k=total_word_num)
            token_lists = list(set(possible_token_ids.cpu().tolist()))

        elif sample_strategy == "predefined_token_list":
            possible_token_ids = predefined_possible_token_ids
            token_lists = list(set(possible_token_ids.cpu().tolist()))
        else:
            print("Unknown sample strategy: ", sample_strategy)
            raise NotImplementedError

        res_matrix_raw = generate_random_choose_probs_dipmark(
            model,
            model_id,
            tokenizer,
            cur_green_list,
            cur_red_list,
            dipmark_alpha,
            ctx_ids,
            token_lists,
            batch_size=64,
            blackbox=blackbox,
            sample_num=10,
        )

        res_matrix = torch.zeros_like(res_matrix_raw).cuda()

        for idx1 in range(len(token_lists)):
            for idx2 in range(idx1 + 1, len(token_lists)):
                s1 = res_matrix_raw[idx1][idx2]
                s1_abs = torch.abs(s1)
                pairwise_score = torch.sign(s1) * torch.minimum(
                    s1_abs / delta_predicted, delta_predicted / s1_abs
                )

                res_matrix[idx1][idx2] = pairwise_score
                res_matrix[idx2][idx1] = -pairwise_score

        total_statistics = torch.zeros((4,)).cuda()  # [tp,tn,fp,fn]
        possible_token_statistics = torch.zeros((4,)).cuda()  # [tp,tn,fp,fn]

        res_matrix_sum = res_matrix.sum(dim=-1)

        token_scores = torch.zeros((len(token_lists),)).cuda()

        for idx1 in range(len(token_lists)):
            for idx2 in range(len(token_lists)):
                token_scores[idx1] += (
                    torch.abs(res_matrix_sum[idx2]) * res_matrix[idx1][idx2]
                )

        final_green_list_tokens = []
        for idx in range(len(token_lists)):
            if token_scores[idx] > 0:
                final_green_list_tokens.append(idx)

        for idx in range(len(token_lists)):
            if idx in final_green_list_tokens:  # predicted as green
                if token_lists[idx] in cur_green_list:  # True positive
                    total_statistics[0] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[0] += 1
                else:  # False positive
                    total_statistics[2] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[2] += 1

            else:  # predicted as red
                if token_lists[idx] in cur_red_list:  # True negative
                    total_statistics[1] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[1] += 1
                else:  # False negative
                    total_statistics[3] += 1
                    if token_lists[idx] in possible_token_ids:
                        possible_token_statistics[3] += 1

        if print_all:
            print("Local all token results:")
            print(cal_results(total_statistics))
            print("Local posible token results:")
            print(cal_results(possible_token_statistics), flush=True)

        detected_green_token_ids = []
        for idx in range(len(token_lists)):
            if (token_lists[idx] in possible_token_ids) and token_scores[idx] > 0:
                detected_green_token_ids.append(token_lists[idx])

        return detected_green_token_ids, total_statistics, possible_token_statistics
