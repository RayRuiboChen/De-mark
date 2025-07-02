import os
import torch
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import tqdm
import random
from nltk.corpus import brown
from nltk import FreqDist
from torch.utils.data import Dataset, DataLoader
import math
from scipy.stats import norm
import random

"""
mannually apply chat template, and add the previous n-grams to output.
(can be also achieved by filter out the outputs that previous n-grams are inconsistent)
"""


class QueryDataset(Dataset):
    def __init__(self, ctx_ids, token_lists, tokenizer, model_id):
        self.data = []
        self.tokenizer = tokenizer
        for idx1 in range(len(token_lists)):
            for idx2 in range(idx1 + 1, len(token_lists)):
                target_ids = [token_lists[idx1], token_lists[idx2]]
                self.data.append(
                    [
                        torch.tensor([idx1, idx2]).long(),
                        self.construct_input_ids(ctx_ids, target_ids, model_id),
                    ]
                )

    def construct_input_ids(self, ctx_ids, target_ids, model_id):
        if (
            model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct"
            or model_id == "meta-llama/Llama-3.2-3B-Instruct"
            or model_id == "meta-llama/Llama-3.2-1B-Instruct"
        ):
            t1 = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>
            
I need you to randomly choose a phrase without exact meaning. Randomly start your answer with: \"'''
            t2 = '" or "'
            t3 = '".<|eot_id|><|start_header_id|>assitant<|end_header_id|>\n\n'

            t1_ids = (
                self.tokenizer(t1, add_special_tokens=False, return_tensors="pt")
                .input_ids[0]
                .cuda()
            )
            t2_ids = (
                self.tokenizer(t2, add_special_tokens=False, return_tensors="pt")
                .input_ids[0]
                .cuda()
            )
            t3_ids = (
                self.tokenizer(t3, add_special_tokens=False, return_tensors="pt")
                .input_ids[0]
                .cuda()
            )

            inputs1 = torch.cat(
                [
                    t1_ids,
                    ctx_ids,
                    torch.tensor([target_ids[0]]).long().cuda(),
                    t2_ids,
                    ctx_ids,
                    torch.tensor([target_ids[1]]).long().cuda(),
                    t3_ids,
                    ctx_ids,
                ],
                dim=0,
            )
            inputs2 = torch.cat(
                [
                    t1_ids,
                    ctx_ids,
                    torch.tensor([target_ids[1]]).long().cuda(),
                    t2_ids,
                    ctx_ids,
                    torch.tensor([target_ids[0]]).long().cuda(),
                    t3_ids,
                    ctx_ids,
                ],
                dim=0,
            )

            return torch.stack([inputs1, inputs2], dim=0)
        elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
            t1 = '''<s>[INST] I need you to randomly choose a phrase without exact meaning. Randomly start your answer with: \"'''
            t2 = '" or "'
            t3 = """\".[/INST] """
            t1_ids = (
                self.tokenizer(t1, add_special_tokens=False, return_tensors="pt")
                .input_ids[0]
                .cuda()
            )
            t2_ids = (
                self.tokenizer(t2, add_special_tokens=False, return_tensors="pt")
                .input_ids[0]
                .cuda()
            )
            t3_ids = (
                self.tokenizer(t3, add_special_tokens=False, return_tensors="pt")
                .input_ids[0]
                .cuda()
            )

            inputs1 = torch.cat(
                [
                    t1_ids,
                    ctx_ids,
                    torch.tensor([target_ids[0]]).long().cuda(),
                    t2_ids,
                    ctx_ids,
                    torch.tensor([target_ids[1]]).long().cuda(),
                    t3_ids,
                    ctx_ids,
                ],
                dim=0,
            )
            inputs2 = torch.cat(
                [
                    t1_ids,
                    ctx_ids,
                    torch.tensor([target_ids[1]]).long().cuda(),
                    t2_ids,
                    ctx_ids,
                    torch.tensor([target_ids[0]]).long().cuda(),
                    t3_ids,
                    ctx_ids,
                ],
                dim=0,
            )

            return torch.stack([inputs1, inputs2], dim=0)
        else:
            print("Unknown model id: ", model_id)
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# [tp,tn,fp,fn]
def cal_results(statistics):
    res_dict = {
        "Precision": (statistics[0] / (statistics[0] + statistics[2])).item(),
        "Recall": (statistics[0] / (statistics[0] + statistics[3])).item(),
        "Accuracy": (
            (statistics[0] + statistics[1])
            / (statistics[0] + statistics[1] + statistics[2] + statistics[3])
        ).item(),
    }

    return res_dict


def detect_wm_KGW(ctx_width, prompt_ids, generated_token_ids, tokenizer):

    num = 0
    for idx in range(len(generated_token_ids)):
        new_prompt_ids = torch.cat(
            [
                prompt_ids,
                torch.tensor(generated_token_ids[:idx], dtype=torch.long)
                .cuda()
                .view(1, -1),
            ],
            dim=1,
        )
        cur_green_list, cur_red_list = get_ngram_red_green_list(
            new_prompt_ids[0, -ctx_width:], tokenizer.vocab_size, new_prompt_ids.device
        )
        if generated_token_ids[idx] in cur_green_list:
            num += 1

    cur_z_score = (
        2 * (num - len(generated_token_ids) / 2) / math.sqrt(len(generated_token_ids))
    )

    print("detected greenlist number:", num)
    print("total generated tokens number:", len(generated_token_ids))
    print("z_score:", cur_z_score)
    print("p:", 1 - norm.cdf(cur_z_score))
    detect_wm_res = {
        "z_score": cur_z_score,
        "p": 1 - norm.cdf(cur_z_score),
        "detected_greenlist_number": num,
        "total_generated_tokens_number": len(generated_token_ids),
    }
    return detect_wm_res


def detect_wm_dipmark(ctx_width, prompt_ids, generated_token_ids, tokenizer):

    num = 0
    for idx in range(len(generated_token_ids)):
        new_prompt_ids = torch.cat(
            [
                prompt_ids,
                torch.tensor(generated_token_ids[:idx], dtype=torch.long)
                .cuda()
                .view(1, -1),
            ],
            dim=1,
        )
        cur_green_list, cur_red_list = get_ngram_red_green_list(
            new_prompt_ids[0, -ctx_width:], tokenizer.vocab_size, new_prompt_ids.device
        )
        if generated_token_ids[idx] in cur_green_list:
            num += 1
    # print('total length:',len(generated_token_ids))
    # print('detected num:',num)

    cur_z_score = (num - len(generated_token_ids) / 2) / math.sqrt(
        len(generated_token_ids)
    )

    print("detected greenlist number:", num)
    print("total generated tokens number:", len(generated_token_ids))
    print("beta_score:", cur_z_score)
    print("p:", math.exp(-2 * cur_z_score**2))

    detect_wm_res = {
        "z_score": cur_z_score,
        "p": math.exp(-2 * cur_z_score**2),
        "detected_greenlist_number": num,
        "total_generated_tokens_number": len(generated_token_ids),
    }
    return detect_wm_res


def generate_random_choose_probs(
    model,
    model_id,
    tokenizer,
    cur_green_list,
    delta,
    ctx_ids,
    token_lists,
    batch_size=4,
    blackbox=False,
    sample_num=100,
):

    # [idx1,idx2]
    probs_res_matrix = torch.zeros((len(token_lists), len(token_lists))).cuda()
    query_dataset = QueryDataset(
        ctx_ids=ctx_ids, token_lists=token_lists, tokenizer=tokenizer, model_id=model_id
    )

    token_lists = torch.tensor(token_lists).cuda().long()
    for token_list_indices, inputs in DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    ):

        cur_bsz, _, seq_len = inputs.shape
        inputs = inputs.reshape(cur_bsz * 2, seq_len)

        forward_res = model(inputs)
        logits = forward_res.logits[:, -1, :]  # [bsz*2,vocab_size]
        logits[:, cur_green_list] += delta
        probs = F.softmax(logits, dim=-1)  # [bsz*2,vocab_size]

        if not blackbox:
            probs = probs.reshape(cur_bsz, 2, -1).sum(dim=1)  # [bsz,vocab_size]

            s1 = torch.log(
                probs[range(cur_bsz), token_lists[token_list_indices[:, 0]]]
                / probs[range(cur_bsz), token_lists[token_list_indices[:, 1]]]
            )

            probs_res_matrix[token_list_indices[:, 0], token_list_indices[:, 1]] = s1
            probs_res_matrix[token_list_indices[:, 1], token_list_indices[:, 0]] = -s1
        else:  # blackbox setting
            probs = probs.reshape(cur_bsz, 2, -1)  # [bsz,2,vocab_size]
            p_1 = probs[
                range(cur_bsz), :, token_lists[token_list_indices[:, 0]]
            ]  # [bsz,2]
            p_2 = probs[
                range(cur_bsz), :, token_lists[token_list_indices[:, 1]]
            ]  # [bsz,2]

            normed_prob = p_1 / (p_1 + p_2)  # [bsz,2]
            random_v = torch.rand((sample_num, cur_bsz, 2)).cuda()

            res = random_v < normed_prob.view(1, cur_bsz, 2)
            res = res.sum(dim=0)
            res = res.sum(dim=-1)

            res = torch.minimum(
                res, torch.full_like(res, 2 * sample_num - 1, device=res.device)
            )
            res = torch.maximum(res, torch.ones_like(res, device=res.device))
            s1 = torch.log(res / (2 * sample_num - res))
            probs_res_matrix[token_list_indices[:, 0], token_list_indices[:, 1]] = s1
            probs_res_matrix[token_list_indices[:, 1], token_list_indices[:, 0]] = -s1
    return probs_res_matrix


def generate_random_choose_probs_dipmark(
    model,
    model_id,
    tokenizer,
    cur_green_list,
    cur_red_list,
    alpha,
    ctx_ids,
    token_lists,
    batch_size=4,
    blackbox=False,
    sample_num=100,
):

    # [idx1,idx2]
    probs_res_matrix = torch.zeros((len(token_lists), len(token_lists))).cuda()
    query_dataset = QueryDataset(
        ctx_ids=ctx_ids, token_lists=token_lists, tokenizer=tokenizer, model_id=model_id
    )
    token_lists = torch.tensor(token_lists).cuda().long()
    for token_list_indices, inputs in DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    ):

        cur_bsz, _, seq_len = inputs.shape
        inputs = inputs.reshape(cur_bsz * 2, seq_len)

        # print(tokenizer.decode(inputs[0]))
        forward_res = model(inputs)
        logits = forward_res.logits[:, -1, :]  # [bsz*2,vocab_size]
        # logits[:,cur_green_list]+=delta
        logits = dipmark_logit_processor(
            logits, greenlist_ids=cur_green_list, redlist_ids=cur_red_list, alpha=alpha
        )
        probs = F.softmax(logits, dim=-1)  # [bsz*2,vocab_size]

        if not blackbox:
            probs = probs.reshape(cur_bsz, 2, -1).sum(dim=1)  # [bsz,vocab_size]
            s1 = torch.log(
                probs[range(cur_bsz), token_lists[token_list_indices[:, 0]]]
                / probs[range(cur_bsz), token_lists[token_list_indices[:, 1]]]
            )

            probs_res_matrix[token_list_indices[:, 0], token_list_indices[:, 1]] = s1
            probs_res_matrix[token_list_indices[:, 1], token_list_indices[:, 0]] = -s1
        else:  # blackbox setting
            probs = probs.reshape(cur_bsz, 2, -1)  # [bsz,2,vocab_size]
            p_1 = probs[
                range(cur_bsz), :, token_lists[token_list_indices[:, 0]]
            ]  # [bsz,2]
            p_2 = probs[
                range(cur_bsz), :, token_lists[token_list_indices[:, 1]]
            ]  # [bsz,2]

            normed_prob = p_1 / (p_1 + p_2)  # [bsz,2]
            random_v = torch.rand((sample_num, cur_bsz, 2)).cuda()

            res = random_v < normed_prob.view(1, cur_bsz, 2)
            res = res.sum(dim=0)
            res = res.sum(dim=-1)

            res = torch.minimum(
                res, torch.full_like(res, 2 * sample_num - 1, device=res.device)
            )
            res = torch.maximum(res, torch.ones_like(res, device=res.device))
            # print(res)
            # print(res.shape)
            s1 = torch.log(res / (2 * sample_num - res))
            probs_res_matrix[token_list_indices[:, 0], token_list_indices[:, 1]] = s1
            probs_res_matrix[token_list_indices[:, 1], token_list_indices[:, 0]] = -s1

    return probs_res_matrix


def generate_random_choose_probs_on_actual_ctx(
    model,
    model_id,
    tokenizer,
    ctx_width,
    delta,
    ctx_ids,
    token_lists,
    batch_size=32,
    blackbox=False,
    sample_num=100,
):
    probs_res_matrix = torch.zeros((len(token_lists), len(token_lists))).cuda()
    query_dataset = QueryDataset(
        model_id=model_id, ctx_ids=ctx_ids, token_lists=token_lists, tokenizer=tokenizer
    )
    for token_list_indices, inputs in tqdm.tqdm(
        DataLoader(query_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    ):

        cur_bsz, _, seq_len = inputs.shape
        inputs = inputs.reshape(cur_bsz * 2, seq_len)

        forward_res = model(inputs)
        logits = forward_res.logits[:, -1, :]  # [bsz*2,vocab_size]

        cur_green_list, _ = get_ngram_red_green_list(
            inputs[0, -ctx_width:], tokenizer.vocab_size, inputs.device
        )
        logits[:, cur_green_list] += delta
        probs = F.softmax(logits, dim=-1)
        if not blackbox:
            probs = probs.reshape(cur_bsz, 2, -1).sum(dim=1)  # [bsz,vocab_size]

            token_lists = token_lists.cuda().long()
            s1 = torch.log(
                probs[range(cur_bsz), token_lists[token_list_indices[:, 0]]]
                / probs[range(cur_bsz), token_lists[token_list_indices[:, 1]]]
            )

            probs_res_matrix[token_list_indices[:, 0], token_list_indices[:, 1]] = s1
            probs_res_matrix[token_list_indices[:, 1], token_list_indices[:, 0]] = -s1
        else:  # blackbox setting
            token_lists = torch.tensor(token_lists).cuda().long()
            probs = probs.reshape(cur_bsz, 2, -1)  # [bsz,2,vocab_size]
            p_1 = probs[
                range(cur_bsz), :, token_lists[token_list_indices[:, 0]]
            ]  # [bsz,2]
            p_2 = probs[
                range(cur_bsz), :, token_lists[token_list_indices[:, 1]]
            ]  # [bsz,2]

            normed_prob = p_1 / (p_1 + p_2)  # [bsz,2]
            random_v = torch.rand((sample_num, cur_bsz, 2)).cuda()

            res = random_v < normed_prob.view(1, cur_bsz, 2)
            res = res.sum(dim=0)
            res = res.sum(dim=-1)

            res = torch.minimum(
                res, torch.full_like(res, 2 * sample_num - 1, device=res.device)
            )
            res = torch.maximum(res, torch.ones_like(res, device=res.device))

            s1 = torch.log(res / (2 * sample_num - res))
            probs_res_matrix[token_list_indices[:, 0], token_list_indices[:, 1]] = s1
            probs_res_matrix[token_list_indices[:, 1], token_list_indices[:, 0]] = -s1

    return probs_res_matrix


def get_ngram_red_green_list(context_tokens, vocab_size, device):
    hash = hashlib.sha256()
    hash.update("private_key".encode("utf-8"))
    hash.update(str(context_tokens.cpu()).encode("utf-8"))
    seed = hash.digest()
    seed = int.from_bytes(seed, "big") % (2**32 - 1)

    gamma = 0.5
    greenlist_size = int(vocab_size * gamma)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    vocab_permutation = torch.randperm(vocab_size, device=device, generator=rng)
    green_list_ids = vocab_permutation[:greenlist_size]
    red_list_ids = vocab_permutation[greenlist_size:]
    return green_list_ids, red_list_ids


def top_k_sampling(logits, top_k=10, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    probs[sorted_indices[top_k:]] = 0.0
    probs /= probs.sum()
    next_token = torch.multinomial(probs, 1)
    return next_token


def get_top_k_indices(logits, k):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    return sorted_indices[:k]


def get_target_word_list(tokenizer, low=200, high=4000, model_id=None):
    frequency_list = FreqDist(i.lower() for i in brown.words())
    word_list = [i[0] for i in frequency_list.most_common()][low:high]

    new_word_list = []

    for word in word_list:
        if len(word) < 3:  # filter simple words
            continue

        if len(tokenizer(" " + word, add_special_tokens=False).input_ids) == 1:
            new_word_list.append(" " + word)

    return new_word_list


def dipmark_logit_processor(p_logits, greenlist_ids, redlist_ids, alpha):
    if len(p_logits.shape) == 1:
        shuffle = torch.cat([redlist_ids, greenlist_ids], dim=-1)
        unshuffle = torch.argsort(shuffle, dim=-1)
    elif len(p_logits.shape) == 2:
        shuffle = torch.cat([redlist_ids, greenlist_ids], dim=-1)
        shuffle = shuffle.unsqueeze(dim=0).repeat(p_logits.shape[0], 1)
        unshuffle = torch.argsort(shuffle, dim=-1)
    else:
        raise NotImplementedError

    s_p_logits = torch.gather(p_logits, -1, shuffle)
    s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
    # normalize the log_cumsum to force the last element to be 0
    s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
    s_cumsum = torch.exp(s_log_cumsum)
    s_p = F.softmax(s_p_logits, dim=-1)

    boundary_1 = torch.argmax((s_cumsum > alpha).to(torch.int), dim=-1, keepdim=True)
    p_boundary_1 = torch.gather(s_p, -1, boundary_1)
    portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - alpha) / p_boundary_1
    portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
    s_all_portion_in_right_1 = (s_cumsum > alpha).type_as(p_logits)
    s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

    boundary_2 = torch.argmax(
        (s_cumsum > (1 - alpha)).to(torch.int), dim=-1, keepdim=True
    )
    p_boundary_2 = torch.gather(s_p, -1, boundary_2)
    portion_in_right_2 = (
        torch.gather(s_cumsum, -1, boundary_2) - (1 - alpha)
    ) / p_boundary_2
    portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
    s_all_portion_in_right_2 = (s_cumsum > (1 - alpha)).type_as(p_logits)
    s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

    s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
    s_shift_logits = torch.log(s_all_portion_in_right)
    shift_logits = torch.gather(s_shift_logits, -1, unshuffle)
    return p_logits + shift_logits
