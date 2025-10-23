import argparse
import json
import os
import re
import pdb
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import multiprocessing
import time
import concurrent.futures
import random
from data.build_data import(
    ChatMLTemplate,
    Llama2Template,
    Llama3Template,
    template_dataset
)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--critic_name",
        type=str,
        default="/root/models/llama3-8b-chat-6-730",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="llama3",
    )
    parser.add_argument(
        "--inference_data",
        type=str,
        default=None,
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--device_list", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    return args

def generate_responses(model, sampling_params, prompt):
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def batch_inference(
    inference_dataset,
    sampling_params,
    batch_size,
    critic_name,
    device_id,
    template
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_id))
    model = LLM(model=critic_name, tensor_parallel_size=1, gpu_memory_utilization=0.5)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = template_dataset(batch_items, template, 'critique')
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        for idx, data in enumerate(batch_items):
            new_data = data.copy()
            new_data["critic_model"] = critic_name
            new_data["critique"] = batch_responses[idx]
            results.append(new_data)
    return results


def critic_pipeline(
    critic_name, 
    temperature, 
    sample_num,
    device_list,
    template,
    inference_data
):

    inference_file = inference_data if inference_data is not None else "false_temp.json"
    with open(inference_file, "r") as g:
        ds = json.load(g)
    total_gpu = len(device_list)
    stop_tokens = ["<|end_of_text|>", "</s>", "<|im_end|>","<|endoftext|>"]
    sampling_params = SamplingParams(max_tokens=1536, temperature=temperature, stop=stop_tokens)
    random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    
    critic_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=total_gpu) as executor:
        futures = [
            executor.submit(batch_inference, inference_dataset[idx], sampling_params, 128, critic_name, device_id, template)
            for idx, device_id in enumerate(device_list)
        ]       
        for future in concurrent.futures.as_completed(futures):
            critc = future.result()
            critic_data = critic_data + critc
    with open("false_temp.json", "w") as g:
        json.dump(critic_data, g, indent=2)


if __name__ == "__main__":
    args = arg_parse()
    critic_name = args.critic_name
    temperature = args.temperature
    sample_num = args.sample_num
    device_list = args.device_list.split(',')
    inference_data = args.inference_data
    assert args.template in ['chatml', 'llama2', 'llama3']
    template_dict = {
        'chatml': ChatMLTemplate(),
        'llama2': Llama2Template(),
        'llama3': Llama3Template()
    }
    template = template_dict[args.template]
    critic_pipeline(critic_name, temperature, sample_num, device_list, template, inference_data)