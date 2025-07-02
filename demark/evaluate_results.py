import os
import json
import numpy as np
import torch
import argparse
import openai
import tqdm


def create_rating_prompt(raw_prompt, response):
    prompt_template = """You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed.
Please rate the response on a scale of 1 to 100 by strictly following this format: "[[rating]]", for example: "Rating: [[51]].

Prompt:\n{}\nEnd of prompt.\n\nResponse:\n{}End of response."""
    return prompt_template.format(raw_prompt.strip(), response.strip())


def query_gpt(client, prompt):
    seed = 42
    model_name = "gpt-3.5-turbo"  # Currently points to gpt-3.5-turbo-0125.
    temperature = 0
    max_tokens = 10
    response = None
    success = 1
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print("-" * 80, flush=True)
        # print(f"Caught error querying with unique idx: {unique_idx}", flush=True)
        print("Error info: ", e, flush=True)
        success = 0

    return success, response


def get_gpt_rating(result_path, api_key):
    with open(result_path, "r") as f:
        lines = f.readlines()

    error_list = []

    save_dir = "/".join(result_path.split("/")[:-1])
    save_path = os.path.join(save_dir, "gpt_scores.jsonl")
    error_save_path = os.path.join(save_dir, "error_list.jsonl")

    client = openai.OpenAI(api_key=api_key)

    for line_idx, line in tqdm.tqdm(enumerate(lines)):
        if len(line) < 3:
            continue
        res_dict = json.loads(line)

        prompt = create_rating_prompt(
            raw_prompt=res_dict["raw_prompt"], response=res_dict["generated_text"]
        )

        for idx in range(10):
            success, response = query_gpt(client, prompt)
            if success:
                output_content = response.choices[0].message.content
                # if line_idx==9:
                #     print(output_content)
                st_idx = output_content.find("[[")
                ed_idx = output_content.find("]]")
                if st_idx == -1:
                    continue
                if ed_idx == -1:
                    continue
                try:
                    rate_score = float(output_content[st_idx + 2 : ed_idx])
                except:
                    continue

                break
        if idx == 9:
            error_list.append(line_idx)
            continue

        score_dict = {
            "result_path": result_path,
            "line_idx": line_idx,
            "rating": rate_score,
        }
        with open(save_path, "a") as f:
            f.write(json.dumps(score_dict) + "\n")
            # json.dump(new_sentences,f)

    print("GPT error list for {}:".format(result_path))
    print(error_list)

    if len(error_list) > 0:
        error_dict = {"result_path": result_path, "error_list": error_list}
        with open(error_save_path, "w") as f:
            f.write(json.dumps(error_dict) + "\n")


def get_combined_results(exp_dir):
    new_filename = "total_results.jsonl"
    res_list = []
    added_prompt_idx_list = []  # avoid duplications

    separate_dir = os.path.join(exp_dir, "separate_results")
    for filename in os.listdir(separate_dir):
        if filename == new_filename:
            continue
        with open(os.path.join(separate_dir, filename), "r") as f:
            lines = f.readlines()

        for line in lines:
            if len(line) < 3:
                continue
            line = line.strip()
            cur_res_dict = json.loads(line)
            cur_prompt_idx = cur_res_dict["prompt_idx"]
            if cur_prompt_idx in added_prompt_idx_list:
                continue

            added_prompt_idx_list.append(cur_prompt_idx)
            res_list.append(cur_res_dict)

    res_list = sorted(res_list, key=lambda x: x["prompt_idx"])

    with open(os.path.join(exp_dir, new_filename), "w") as f:
        for i in res_list:
            f.write(json.dumps(i) + "\n")


def get_p_list(jsonl_file):
    with open(jsonl_file, "r") as f:
        lines = f.readlines()

    p_list = []
    for line in lines:
        if len(line) < 3:
            continue
        line = line.strip()
        res_dict = json.loads(line)
        dection_res = res_dict["detection_result"]
        p_list.append(dection_res["p"])

    return p_list


def get_avg_rating(rating_path):
    with open(rating_path, "r") as f:
        lines = f.readlines()

    ratings = []
    for line in lines:
        if len(line) < 3:
            continue
        score_dict = json.loads(line)
        ratings.append(score_dict["rating"])

    print("rating path:", rating_path)
    print("average rating:")
    print(sum(ratings) / len(ratings))
    return sum(ratings) / len(ratings)


def get_statistics(total_res_path):
    with open(total_res_path, "r") as f:
        lines = f.readlines()

    total_stat = torch.zeros((4,))
    for line in lines:
        if len(line) < 3:
            continue

        res_dict = json.loads(line)

        stat = res_dict["possible_token_statistics"]
        total_stat += torch.tensor(stat)

    precision = total_stat[0] / (total_stat[0] + total_stat[2])
    recall = total_stat[0] / (total_stat[0] + total_stat[3])
    f1 = 2 * precision * recall / (precision + recall)
    acc = (total_stat[0] + total_stat[1]) / torch.sum(total_stat)

    stat_res = {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "accuracy": acc.item(),
    }
    return stat_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--openai_api_key", type=str)

    args = parser.parse_args()
    exp_dir = args.exp_dir
    fpr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    total_path = os.path.join(exp_dir, "total_results.jsonl")
    gpt_score_path = os.path.join(exp_dir, "gpt_scores.jsonl")
    evaluation_path = os.path.join(exp_dir, "evaluation_results.json")

    if not os.path.exists(total_path):
        get_combined_results(exp_dir=exp_dir)

    if not os.path.exists(evaluation_path):
        print("getting gpt rating...")
        if not os.path.exists(gpt_score_path):
            get_gpt_rating(result_path=total_path, api_key=args.openai_api_key)
        avg_rating = get_avg_rating(rating_path=gpt_score_path)

        p_list = get_p_list(jsonl_file=os.path.join(exp_dir, "total_results.jsonl"))
        p_list = np.array(p_list)

        evaluation_dict = {"Avg_GPT_rating": avg_rating, "median_p": np.median(p_list)}

        stat_res = get_statistics(total_path)
        evaluation_dict["stat_res"] = stat_res

        for fpr in fpr_list:
            evaluation_dict["TPR@FPR={}".format(str(fpr))] = np.mean(p_list < fpr)

        with open(evaluation_path, "w") as f:
            json.dump(evaluation_dict, f)

    with open(evaluation_path, "r") as f:
        evaluation_dict = json.load(f)
    print("evaluation results:", evaluation_dict)


if __name__ == "__main__":
    main()
