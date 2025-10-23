import argparse
import json
import os
import re
import pdb
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing
import time
from process_ans.eval_utils import extract_answer, compare_answer_with_groundtruth
from collections import Counter, defaultdict
import pandas as pd

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs='+',
        default=["openai/gsm8k"],
    )
    parser.add_argument(
        "--dataset_type",
        nargs='+',
        default=["gsm8k"],
    )
    parser.add_argument("--results_file", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct-selfimprove.json")
    parser.add_argument("--all_refinement_file", type=str, default="all_refinement.json")
    parser.add_argument("--mode", type=str, default="passk")
    args = parser.parse_args()
    return args
    

def safe_eval(expr):
    # 只允许数字和基本运算符
    if re.match(r'^[\d\s+\-*/().]*$', expr):
        return eval(expr)
    else:
        raise ValueError("not an equation or number")


def sequential_data_filter():
    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    for data in ds:
        data["origin_response"] = data["new_response"]
        data["correctness-origin"] = data["correctness-new"]
        del data["critique"]
        del data["critic_model"]
        del data["new_response"]
        del data["correctness-new"]
    with open("false_temp.json", "w") as g:
        json.dump(ds, g, indent=2)

def only_final_sequential(results_file):
    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    with open(results_file, "w") as f:
        for data in ds:
            f.write(json.dumps(data))
            f.write("\n")

def response_extracter(ds, results_file):
    response_bucket = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        response_bucket[key] = []
    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if "new_response" in data.keys():
                response = data["new_response"]
            else:
                response = data["origin_response"]
            response = extract_answer(response, data["dataset_type"])
            key = data["dataset_type"] + str(data["idx"])
            response_bucket[key].append(response)
    return response_bucket
    
def pass_at_k(ds, response_bucket):
    accuracy = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        if data["dataset_type"] not in accuracy.keys():
            accuracy[data["dataset_type"]] = 0
        if key in response_bucket.keys():
            for item in response_bucket[key]:
                gt_ans = extract_answer(data["answer"], dataset_name=data["dataset_type"])
                flag1 = compare_answer_with_groundtruth(gt_ans, item)
                if flag1 == True:
                    accuracy[data["dataset_type"]] += 1
                    break
    print("passk")
    return accuracy

def avg_at_k(ds, response_bucket):
    accuracy = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        if data["dataset_type"] not in accuracy.keys():
            accuracy[data["dataset_type"]] = 0
        if key in response_bucket.keys():
            for item in response_bucket[key]:
                gt_ans = extract_answer(data["answer"], dataset_name=data["dataset_type"])
                flag1 = compare_answer_with_groundtruth(gt_ans, item)
                if flag1 == True:
                    accuracy[data["dataset_type"]] += 1
    print("avgk")
    return accuracy

def majority_vote(ds, response_bucket):
    accuracy = {}
    for data in ds:
        key = data["dataset_type"] + str(data["idx"])
        if data["dataset_type"] not in accuracy.keys():
            accuracy[data["dataset_type"]] = 0
        mv_response = Counter(response_bucket[key]).most_common(1)[0][0]
        gt_ans = extract_answer(data["answer"], dataset_name=data["dataset_type"])
        flag1 = compare_answer_with_groundtruth(gt_ans, mv_response)
        if flag1 == True:
            accuracy[data["dataset_type"]] += 1
    return accuracy

def accuracy_filter(accuracy):
    if "math" in accuracy.keys():
        accuracy["math"] = accuracy["math"] / 5000
    if "gsm8k" in accuracy.keys():
        accuracy["gsm8k"] = accuracy["gsm8k"] / 1319
    if "aqua" in accuracy.keys():
        accuracy["aqua"] = accuracy["aqua"] / 254
    if "svamp" in accuracy.keys():
        accuracy["svamp"] = accuracy["svamp"] / 300
    if "theoremqa" in accuracy.keys():
        accuracy["theoremqa"] = accuracy["theoremqa"] / 800
    return accuracy

def test_file_sort(results_file):
    ds = []
    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line)
            ds.append(data)
    ds_list = sorted(ds, key=lambda x: (x['dataset_type'], x['idx']))

    with open(results_file, "w") as f:
        for data in ds_list:
            f.write(json.dumps(data))
            f.write("\n")

def correctness_filter(correctness, data):
    return {
        k: data[k] / correctness[k] if correctness[k]!=0 else 0  for k in correctness.keys()
    }


def calculate_x2x(all_refinement_file):
    with open(all_refinement_file, "r") as f:
        datas = json.load(f)
    c2c = {}
    c2i = {}
    i2i = {}
    i2c = {}
    origin_correct = {}
    refine_correct = {}
    for data in datas:
        dataset_type = data["dataset_type"]
        if dataset_type not in c2c.keys():
            c2c[dataset_type] = 0
            c2i[dataset_type] = 0
            i2c[dataset_type] = 0
            i2i[dataset_type] = 0
            origin_correct[dataset_type] = 0
            refine_correct[dataset_type] = 0
        if data["correctness-origin"]:
            origin_correct[dataset_type] += 1
        if data["correctness-new"]:
            refine_correct[dataset_type] += 1
        if data["correctness-origin"] and data["correctness-new"]:
            c2c[dataset_type] += 1
        elif data["correctness-origin"] and not data["correctness-new"]:
            c2i[dataset_type] += 1
        elif not data["correctness-origin"] and data["correctness-new"]:
            i2c[dataset_type] += 1
        elif not data["correctness-origin"] and not data["correctness-new"]:
            i2i[dataset_type] += 1
    c2c_result = accuracy_filter(c2c)
    c2c_result["type"] = "c2c"
    c2i_result = accuracy_filter(c2i)
    c2i_result["type"] = "c2i"
    i2c_result = accuracy_filter(i2c)
    i2c_result["type"] = "i2c"
    i2i_result = accuracy_filter(i2i)
    i2i_result["type"] = "i2i"
    delta = {
        k: refine_correct[k] - origin_correct[k] for k in origin_correct.keys()
    }
    delta_result = accuracy_filter(delta)
    delta_result["type"] = "delta"
    final_result = [c2c_result, c2i_result, i2c_result, i2i_result, delta_result]
    return final_result

def calculate_discrimination_relevance(all_refinement_file, columns):
    with open(all_refinement_file, "r") as f:
        datas = json.load(f)
    discrimination_c = defaultdict(int)
    discrimination_i = defaultdict(int)
    discrimination = defaultdict(int)
    relevance_c2c = defaultdict(int)
    relevance_c2i = defaultdict(int)
    relevance_i2c = defaultdict(int)
    relevance_i2i = defaultdict(int)
    relevance = defaultdict(int)
    origin_correct = defaultdict(int)
    origin_wrong = defaultdict(int)
    c2c_num = defaultdict(int)
    c2i_num = defaultdict(int)
    i2c_num = defaultdict(int)
    i2i_num = defaultdict(int)
    for data in datas:
        dataset_type = data["dataset_type"]
        correctness_origin = data["correctness-origin"]
        correctness_new = data["correctness-new"]
        critique = data["critique"]
        correct_critique_result = "Correctness of the final answer: Correct"
        wrong_critique_result = "Correctness of the final answer: Wrong"
        if correctness_origin:
            if correct_critique_result in critique:
                discrimination[dataset_type] += 1
                discrimination_c[dataset_type] += 1
                critique_result = "c2c"
            elif wrong_critique_result in critique:
                critique_result = "c2i"
            else:
                continue
            origin_correct[dataset_type] += 1
            if correctness_new:
                c2c_num[dataset_type] += 1
            else:
                c2i_num[dataset_type] += 1
        else:
            if correct_critique_result in critique:
                critique_result = "i2i"
            elif wrong_critique_result in critique:
                discrimination[dataset_type] += 1
                discrimination_i[dataset_type] += 1
                critique_result = "i2c"
            else:
                continue
            origin_wrong[dataset_type] += 1
            if correctness_new:
                i2c_num[dataset_type] += 1
            else:
                i2i_num[dataset_type] += 1
        if critique_result == "c2c" and correctness_origin and correctness_new:
            relevance_c2c[dataset_type] += 1
        elif critique_result == "c2i" and correctness_origin and not correctness_new:
            relevance_c2i[dataset_type] += 1
        elif critique_result == "i2c" and not correctness_origin and correctness_new:
            relevance_i2c[dataset_type] += 1
        elif critique_result == "i2i" and not correctness_origin and not correctness_new:
            relevance_i2i[dataset_type] += 1
    relevance = {
        k: (relevance_c2c[k] + relevance_i2c[k]) / (discrimination[k]) for k in discrimination.keys()
    }
    relevance_i = {
        k: (relevance_i2c[k]) / (discrimination_i[k]) for k in discrimination.keys()
    }
    relevance_c = {
        k: relevance_c2c[k] / discrimination_c[k] if discrimination_c[k] != 0 else 0 for k in discrimination.keys()
    }
    discrimination_len = {
        k: origin_correct[k] + origin_wrong[k] for k in discrimination.keys()
    }
    discrimination_result = correctness_filter(discrimination_len, discrimination)
    discrimination_result["type"] = "disc"
    discrimination_c_result = correctness_filter(origin_correct, discrimination_c)
    discrimination_c_result["type"] = "disc_c"
    discrimination_i_result = correctness_filter(origin_wrong, discrimination_i)
    discrimination_i_result["type"] = "disc_i"
    relevance["type"] = "relev"
    relevance_i["type"] = "relev_i"
    relevance_c["type"] = "relev_c"
    print(pd.DataFrame([discrimination_c_result, discrimination_i_result, discrimination_result, relevance_c, relevance_i , relevance], columns=columns))


if __name__ == "__main__":
    args = arg_parse()
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    results_file = args.results_file
    mode = args.mode
    all_refinement_file = args.all_refinement_file
    if mode == "sequential":
        sequential_data_filter()
    elif mode == "only_final_sequential":
        only_final_sequential(results_file)
    else:
        ds = []
        for ds_name, ds_type in zip(dataset_name, dataset_type):
            if ds_type == "gsm8k":
                dataset = load_dataset(ds_name, "main", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "math":
                dataset = load_dataset(ds_name, "all", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("problem")
                    data["answer"] = data.pop("solution")
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "aqua":
                with open(f"{ds_name}/test.json", "r") as f:
                    dataset = json.load(f)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "mathqa":
                dataset = load_dataset(ds_name, "all", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    data["question"] = f"{data.pop('Problem')}\nAnswer Choices: {data.pop('options')}"
                    data["answer"] = data.pop("Rationale")
                    ds.append(data)
            if ds_type == "svamp":
                dataset = load_dataset(ds_name, "default", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("question_concat")
                    data["answer"] = data.pop("Equation") + "=" + data.pop("Answer")
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "theoremqa":
                dataset = load_dataset(ds_name, "default", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("Question")
                    data["answer"] = data.pop("Answer")
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "omnimath":
                dataset = load_dataset(ds_name, "default", split = "test", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("problem")
                    data["answer"] = data.pop("answer")
                    data["dataset_type"] = ds_type
                    ds.append(data)
        response_bucket = response_extracter(ds, results_file)
        if mode == "passk":
            accuracy = pass_at_k(ds, response_bucket)
        elif mode == "avgk":
            accuracy = avg_at_k(ds, response_bucket)
        elif mode == "majority":
            accuracy = majority_vote(ds, response_bucket)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', 100)
        columns = ["type"] + dataset_type
        accuracy = accuracy_filter(accuracy)
        accuracy["type"] = "acc"
        metrics = calculate_x2x(all_refinement_file)
        metrics.append(accuracy)
        print(pd.DataFrame(metrics, columns=columns))
        calculate_discrimination_relevance(all_refinement_file, columns)
        test_file_sort(results_file)
