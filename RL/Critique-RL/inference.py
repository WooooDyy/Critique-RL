import argparse
import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import multiprocessing
import time
from process_ans.utils import compare_answer
from data.build_data import(
    DefaultTemplate,
    ChatMLTemplate,
    Llama2Template,
    Llama3Template,
    template_dataset
)
import random

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppo_prompt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--template",
        type=str,
        default="llama3",
    )
    parser.add_argument(
        "--actor_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--results_file", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct-selfimprove.json")
    parser.add_argument("--mode", type=str, default="inference")
    parser.add_argument("--test_know_answer", type=int, default=1)
    parser.add_argument("--need_false_data", type=int, default=0)
    parser.add_argument("--reserved_new_data", type=int, default=1)
    parser.add_argument("--device_list", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    return args


def generate_responses(model, sampling_params, prompt):
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def batch_inference(
    inference_dataset,
    device_id,
    sampling_params,
    batch_size,
    actor_name,
    template
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results_true = []
    results_false = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = template_dataset(batch_items, template, 'inference')
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            question = data["question"]
            answer = data["answer"]
            response = batch_responses[idx]
            correct = compare_answer(answer, response, data["dataset_type"])
            new_data = {}
            new_data["idx"] = data["idx"]
            new_data["dataset_type"] = data["dataset_type"]
            new_data["question"] = question
            new_data["answer"] = answer
            new_data["actor_model"] = actor_name
            new_data["origin_response"] = "Let's break it down step by step:\n\n" + response
            new_data["correctness-origin"] = correct
            if correct == True:
                results_true.append(new_data)
            else:
                results_false.append(new_data)
    return results_true, results_false


def inference_pipeline(
    actor_name,
    ds, 
    temperature, 
    sample_num, 
    results_path,
    device_list,
    template, 
    ppo_prompt_path=None
):
    total_gpu = len(device_list)
    stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant", "<|im_end|>","<|endoftext|>"]
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    true_data = []
    false_data = []
    all_data = []

    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_inference, args=(inference_dataset[idx], device_id, sampling_params, 128, actor_name, template))
            for idx, device_id in enumerate(device_list)
        ]
        for r in results:
            result_true, result_false = r.get()
            true_data += result_true
            false_data += result_false
    with open(results_path, "a+") as g:
        for new_data in true_data:
            g.write(json.dumps(new_data))
            g.write("\n")
    with open("false_temp.json", "w") as g:
        json.dump(false_data , g, indent=2)
    with open("all_inference.json", "w") as g:
        json.dump(false_data + true_data, g, indent=2)
    if ppo_prompt_path is not None:
        with open(f"{ppo_prompt_path}/false_temp.json", "w") as g:
            json.dump(false_data , g, indent=2)
        with open(f"{ppo_prompt_path}/all_inference.json", "w") as g:
            json.dump(false_data + true_data, g, indent=2)

def shortcut_barrier(answer, origin_response, new_response):
    numbers_answer = len(re.findall(r'\d+', answer))
    numbers_origin = len(re.findall(r'\d+', origin_response))
    numbers_new = len(re.findall(r'\d+', new_response))
    if numbers_new < min(numbers_answer, numbers_origin) - 2:
        return False
    else:
        return True

def batch_new(
    inference_dataset,
    sampling_params,
    batch_size,
    actor_name,
    device_id,
    reserved_new_data,
    template
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results = []
    false_results = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = template_dataset(batch_items, template, 'refinement')
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            new_response = batch_responses[idx]
            answer = data["answer"]
            true_flag = False
            correct = compare_answer(answer, new_response, data["dataset_type"])
            if correct == True:
                if len(results) <= (reserved_new_data-1) or (len(results) > (reserved_new_data-1) and data["critique"] != results[-reserved_new_data]["critique"]):
                    if shortcut_barrier(data["answer"], data["origin_response"], new_response):
                        new_data = data.copy()
                        new_data["new_response"] = "Let's break it down step by step:\n\n" + new_response
                        new_data["correctness-new"] = correct
                        results.append(new_data)
                        true_flag = True
            if not true_flag:
                new_data = data.copy()
                new_data["new_response"] = "Let's break it down step by step:\n\n" + new_response
                new_data["correctness-new"] = correct
                false_results.append(new_data)
    return results, false_results


def new_pipeline(
    actor_name,
    temperature,
    sample_num,
    results_file,
    reserved_new_data,
    device_list,
    template,
    need_false_data = 0,
    ppo_prompt_path = None
):
    with open("false_temp.json", "r") as g:
        ds = json.load(g)
    total_gpu = len(device_list)
    stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant", "<|im_end|>","<|endoftext|>"]
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])

    new_data = []
    false_new_data = []
    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_new, args=(inference_dataset[idx], sampling_params, 128, actor_name, device_id, reserved_new_data, template))
            for idx, device_id in enumerate(device_list)
        ]
        for r in results:
            new, false_new = r.get()
            new_data = new_data + new
            false_new_data = false_new_data + false_new
    with open("false_temp.json", "w") as g:
        json.dump(new_data + false_new_data, g, indent=2)
    with open("all_refinement.json", "w") as g:
        json.dump(new_data + false_new_data, g, indent=2)
    if ppo_prompt_path is not None:
        with open(f"{ppo_prompt_path}/all_refinement.json", "w") as g:
            json.dump(new_data + false_new_data, g, indent=2)
    with open(results_file, "a+") as g:
        for da in new_data:
            g.write(json.dumps(da))
            g.write("\n")
    if need_false_data:
        with open(results_file, "a+") as g:
            for da in false_new_data:
                g.write(json.dumps(da))
                g.write("\n") 

def batch_test(
    inference_dataset,
    device_id,
    sampling_params,
    batch_size,
    actor_name,
    template
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results_true = []
    results_false = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = template_dataset(batch_items, template, 'inference')
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            question = data["question"]
            answer = data["answer"]
            response = batch_responses[idx]
            correct = compare_answer(answer, response, data["dataset_type"])
            new_data = {}
            new_data["idx"] = data["idx"]
            new_data["dataset_type"] = data["dataset_type"]
            new_data["question"] = question
            new_data["answer"] = answer
            new_data["actor_model"] = actor_name
            new_data["origin_response"] = "Let's break it down step by step:\n\n" + response
            new_data["correctness-origin"] = correct
            if correct == True:
                results_true.append(new_data)
            else:
                results_false.append(new_data)
    return results_true, results_false


def test_pipeline(
    actor_name,
    ds, 
    temperature, 
    sample_num, 
    results_path,
    device_list,
    template,
    test_know_answer = 1,
    need_false_data = 0,
):
    total_gpu = len(device_list)
    stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant", "<|im_end|>","<|endoftext|>"]
    sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    true_data = []
    false_data = []

    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_test, args=(inference_dataset[idx], device_id, sampling_params, 128, actor_name, template))
            for idx, device_id in enumerate(device_list)
        ]
        for r in results:
            result_true, result_false = r.get()
            true_data += result_true
            false_data += result_false
    if test_know_answer:
        print("Testing with know answer")
        with open(results_path, "a+") as g:
            for new_data in true_data:
                g.write(json.dumps(new_data))
                g.write("\n")
        with open("false_temp.json", "w") as g:
            json.dump(false_data, g, indent=2)
    else:
        print("Testing with unknown answer")
        false_data += true_data
        with open("false_temp.json", "w") as g:
            json.dump(false_data, g, indent=2)
        if need_false_data:
            with open(results_path, "a+") as g:
                for new_data in false_data:
                    g.write(json.dumps(new_data))
                    g.write("\n")



if __name__ == "__main__":
    args = arg_parse()
    actor_name = args.actor_name
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    temperature = args.temperature
    sample_num = args.sample_num
    results_file = args.results_file
    mode = args.mode
    test_know_answer = args.test_know_answer
    need_false_data = args.need_false_data
    reserved_new_data = args.reserved_new_data
    device_list = args.device_list.split(',')
    assert args.template in ['default', 'chatml', 'llama2', 'llama3']
    template_dict = {
        "default": DefaultTemplate("<|endoftext|>"),  # TODO
        'chatml': ChatMLTemplate(),
        'llama2': Llama2Template(),
        'llama3': Llama3Template()
    }
    template = template_dict[args.template]
    if mode == "inference":
        ds = []
        for ds_name, ds_type in zip(dataset_name, dataset_type):
            if ds_type == "gsm8k":
                dataset = load_dataset(ds_name, "main", split = "train", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "math":
                dataset = load_dataset(ds_name, "all", split = "train", trust_remote_code=True)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["question"] = data.pop("problem")
                    data["answer"] = data.pop("solution")
                    data["dataset_type"] = ds_type
                    ds.append(data)
            if ds_type == "aqua":
                with open(f"{ds_name}/train.json", "r") as f:
                    dataset = json.load(f)
                for idx, data in enumerate(dataset):
                    data["idx"] = idx
                    data["dataset_type"] = ds_type
                    ds.append(data)
        inference_pipeline(actor_name, ds, temperature, sample_num, results_file, device_list, template, ppo_prompt_path=args.ppo_prompt_path)
    elif mode == "new":
        new_pipeline(actor_name, temperature, sample_num, results_file, reserved_new_data, device_list, template, need_false_data, ppo_prompt_path=args.ppo_prompt_path)
    elif mode == "test":
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
        test_pipeline(actor_name, ds, temperature, sample_num, results_file, device_list, template, test_know_answer, need_false_data)
