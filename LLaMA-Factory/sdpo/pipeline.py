from openai import OpenAI
import json
import pdb
from datasets import load_dataset
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import multiprocessing

# model_name = "meta-math/MetaMath-Mistral-7B"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

client2 = OpenAI(
    api_key="",
    base_url="https://api3.apifans.com/v1"
)

def extract_answer(text):
    # 使用正则表达式匹配格式为 "#### 72" 的答案
    match = re.search(r'#### [^\d]*([\d.]+)', text)
    if match:
        return match.group(1)
    else:
        return None
    
def extract_response(text):
    # 使用正则表达式匹配格式为 "#### 72" 的答案
    # match = re.search(r'The answer is: (\d+)', text)
    text.replace(":", "")
    temp = text.find("The answer is", 4)
    text = text[temp:]
    match = re.search(r"The answer is [^\d]*([\d.,]+)", text)
    if match:
        temp = match.group(1)
        if temp[-1] == ".":
            temp = temp[0:-1]
        temp = temp.replace(",", "")
        return temp
    else:
        return None

def compare_answer(answer, response):
    an = extract_answer(answer)
    re = extract_response(response)
    flag = False
    temp = True
    if an is not None and re is not None:
        try:
            if eval(an) == eval(re):
                flag = True
        except Exception as e:
            if an == re:
                flag = True
    if flag == False:
        temp = False
        temp_response = response[response.find("The answer"):]
        if an in temp_response:
            temp = True
    return flag, temp

def generate_raw_template(instruction):
    # prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step.\n"
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is [your answer].'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
    return prompt

def generate_new_template(instruction, raw, critic):
    # prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is [your answer].'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI have made a critic for your reasoning. Please read the following critic and generate a new answer. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is [your answer].'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI have made a step-by-step critic for your solution. Please read the following critic and generate a new answer to correct your solution. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
    return prompt

def generate_new_test_template(instruction, raw, critic):
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction} Please also give your final number in the format of 'The answer is [your answer].'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYou have a wrong step in your reasoning. Please read the following critic and generate a new answer. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critic}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's break it down step by step:\n\n"
    return prompt

def critic_template(question, solution1, solution2):
#     prompt = f"""### Instruction:
# For a given question, there are two corresponding step-by-step solutions. The first solution is a correct one. Please check whether the second solution is correct or wrong. If it is wrong, please then point out and repeat which step in the solution is wrong and give an explanation on why this particular step is incorrect. If it is correct, give a brief explanation on the solution. You don't need to generate the whole correct solution.

# Your answer format should be:
# Correctness of the solution: [Correct/Wrong]
# [If the solution is wrong:]
# Incorrect step sentence: [Repeated incorrect sentence]
# Explanation: [Your explanation]
# [If the solution is correct:]
# Explanation: [Your explanation]

# ### Question:
# {question}

# ### First solution
# {solution1}

# ### Second solution
# {solution2}"""

    prompt = f"""Instruction:
For a given question, there is a step-by-step solution, where each line is a step, and a reference correct answer. Please refer to the correct answer and check step by step to tell me whether the solution is correct or wrong. You should repeat each step and give me an explanation on whether this step is correct or wrong. Finally, you should tell me whether the final answer is correct or wrong. But don't mention anything about the reference answer in your explanation.

Your answer format should be:

Step sentence: [Repeated sentence, ignoring new line break]
Correctness of the step: [Correct/Wrong]
Explanation: [Your Explanation]

Correctness of the final answer: [Correct/Wrong]
Explanation: [Your Explanation]

Question:
{question}

Solution:
{solution2}

Reference answer:
{solution1}"""
    
    return prompt

def critic_template_test(question, solution):
#     prompt = f"""### Instruction:
# For a given question, there is a step-by-step solution. Please check whether the solution is correct or wrong. If it is wrong, please then point out and repeat which step in the solution is wrong and give an explanation on why this particular step is incorrect. If it is correct, give a brief explanation on the solution.

# Your answer format should be:
# Correctness of the solution: [Correct/Wrong]
# [If the solution is wrong:]
# Incorrect step sentence: [Repeated incorrect sentence]
# Explanation: [Your explanation]
# [If the solution is correct:]
# Explanation: [Your explanation]

# ### Question:
# {question}

# ### Solution:
# {solution}"""

#     prompt = f"""### Instruction:
# For a given question, there is a step-by-step solution. Please check the solution step by step and check whether the final answer is correct or wrong. If you find a wrong step, please repeat which step in the solution is wrong and give an explanation on why this particular step is incorrect. If all the steps are correct, give a brief explanation on the solution. Finally, tell me whether the final answer is correct or wrong.

# Your answer format should be:
# [If you find a wrong step:]
# Incorrect step sentence: [Repeated incorrect sentence]
# Explanation: [Your explanation]
# [If all the steps are correct:]
# Explanation: [Your explanation]
# [Correctness of the final answer:]
# Correctness of the answer: [Correct/Wrong]

# ### Question:
# {question}

# ### Solution:
# {solution}"""

    if "Let's break it down step by step:\n\n" in solution:
        solution = solution[len("Let's break it down step by step:\n\n"):]
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nInstruction:
For a given question, there is a step-by-step solution, where each line is a step. Please check step by step to tell me whether the solution is correct or wrong. You should repeat each step and give me an explanation on whether this step is correct or wrong. Finally, you should tell me whether the final answer is correct or wrong.

Your answer format should be:

Step sentence: [Repeated sentence, ignoring new line break]
Correctness of the step: [Correct/Wrong]
Explanation: [Your Explanation]

Correctness of the final answer: [Correct/Wrong]
Explanation: [Your Explanation]

Question:
{question}

Solution:
{solution}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    
    return prompt

def critic_template_base_test(question, solution):
#     instruction = f"""### Instruction:
# For a given question, there is a step-by-step solution. Please check whether the solution is correct or wrong. If it is wrong, please then point out and repeat which step in the solution is wrong and give an explanation on why this particular step is incorrect. If it is correct, give a brief explanation on the solution.

# Your answer format should be:
# Correctness of the solution: [Correct/Wrong]
# [If the solution is wrong:]
# Incorrect step sentence: [Repeated incorrect sentence]
# Explanation: [Your explanation]
# [If the solution is correct:]
# Explanation: [Your explanation]

# ### Question:
# {question}

# ### Solution:
# {solution}
# """
#     instruction = f"""### Instruction:
# For a given question, there is a step-by-step solution. Please check the solution step by step and check whether the final answer is correct or wrong. If you find a wrong step, please repeat which step in the solution is wrong and give an explanation on why this particular step is incorrect. If all the steps are correct, give a brief explanation on the solution. Finally, tell me whether the final answer is correct or wrong.

# Your answer format should be:
# [If you find a wrong step:]
# Incorrect step sentence: [Repeated incorrect sentence]
# Explanation: [Your explanation]
# [If all the steps are correct:]
# Explanation: [Your explanation]
# [Correctness of the final answer:]
# Correctness of the answer: [Correct/Wrong]

# ### Question:
# {question}

# ### Solution:
# {solution}
# """

    if "Let's break it down step by step:\n\n" in solution:
        solution = solution[len("Let's break it down step by step:\n\n"):]
    instruction = f"""Human: Instruction:
For a given question, there is a step-by-step solution, where each line is a step. Please check step by step to tell me whether the solution is correct or wrong. You should repeat each step and give me an explanation on whether this step is correct or wrong. Finally, you should tell me whether the final answer is correct or wrong.

Your answer format should be:

Step sentence: [Repeated sentence, ignoring new line break]
Correctness of the step: [Correct/Wrong]
Explanation: [Your Explanation]

Correctness of the final answer: [Correct/Wrong]
Explanation: [Your Explanation]


Question:
{question}

Solution:
{solution}
Assistant:"""
    return instruction

def generate_critic(text):
    completion = client2.chat.completions.create(
        messages=[
        {"role": "user", "content": text}
    ],
        model="gpt-4o",
        temperature = 0,
    )

    message = completion.choices[0].message.content
    return message


def generate_responses(model, sampling_params, prompt):
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def batch_inference(
    inference_dataset,
    model,
    sampling_params,
    batch_size,
):
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = [
            item
            for item in batch_items
        ]
        # Generate responses using llama
        batch_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        results = results + batch_responses
    return results


def generate_sdpo_pipeline():
    ds = load_dataset("openai/gsm8k", "main", split = "train")
    model = LLM(model=model_name, tensor_parallel_size=4)
    results = []
    inference_dataset = [
        generate_raw_template(item["question"])
        for item in ds
    ]
    for temperature in [0]:
        sampling_params = SamplingParams(max_tokens=1024, temperature=temperature)
        temp_results = batch_inference(inference_dataset, model, sampling_params, 256)
        results = results + temp_results
    for i, data in enumerate(ds):
        question = data["question"]
        answer = data["answer"]
        response = "Let's break it down step by step:\n\n" + results[i]
        flag1, flag2 = compare_answer(answer, response)
        new_data = {}
        new_data["idx"] = i
        new_data["question"] = question
        new_data["answer"] = answer
        new_data["actor_model"] = model_name
        new_data["critic_model"] = "gpt-4o"
        new_data["generate_method"] = "sdpo"
        new_data["origin_response"] = response
        new_data["correctness-origin"] = flag1
        with open(f"{model_name}-gsm8k-sdpo_data_raw(1).json", "a+") as g:
            g.write(json.dumps(new_data))
            g.write("\n")


def generate_critic_pipeline():
    with open(f"{model_name}-gsm8k-sdpo_data_raw(1).json", "r") as f:
        for line in f:
            data = json.loads(line)
            i = data["idx"]
            if i <= -1:
                continue
            question = data["question"]
            answer = data["answer"]
            sdpo_response = data["origin_response"]
            sdpo_response = sdpo_response[len("Let's break it down step by step:\n\n"):]
            critic_prompt = critic_template(question, answer, sdpo_response)
            critic = generate_critic(critic_prompt)
            data["critic"] = critic
            with open(f"{model_name}-gsm8k-sdpo_data_critic(1).json", "a+") as g:
                g.write(json.dumps(data))
                g.write("\n")


def generate_new_pipieline():
    ds = []
    model = LLM(model=model_name, tensor_parallel_size=4)
    results = []
    with open(f"{model_name}-gsm8k-sdpo_data_critic2.json", "r") as f:
        for line in f:
            data = json.loads(line)
            ds.append(data)
    filter_dataset = [
        item
        for item in ds
        if item["correctness-origin"] == False
    ]
    inference_dataset = [
        generate_new_template(item["question"], item["origin_response"], item["critic"])
        for item in ds
        if item["correctness-origin"] == False
    ]
    sampling_params = SamplingParams(max_tokens=1024, temperature=0)
    results = batch_inference(inference_dataset, model, sampling_params, 256)

    for i, data in enumerate(filter_dataset):
        idx = data["idx"]
        if idx <= -1:
            continue
        if idx >= 8000:
            break
        answer = data["answer"]
        new_response = results[i]
        data["new_response"] = "Let's break it down step by step:\n\n" + new_response
        flag1, flag2 = compare_answer(answer, new_response)
        data["correctness-new"] = flag1
        with open(f"{model_name}-gsm8k-sdpo_data(1).json", "a+") as g:
            g.write(json.dumps(data))
            g.write("\n")


def generate_critic_test_pipeline():
    ds = []
    model = LLM(model="/root/models/llama3-8b-8-865", tensor_parallel_size=4)
    results = []
    with open(f"{model_name}-gsm8k-test-sdpo_data_raw.json", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["correctness-origin"] == False:
                ds.append(data)
    inference_dataset = [
        critic_template_base_test(item["question"], item["origin_response"])
        for item in ds
    ]
    sampling_params = SamplingParams(max_tokens=2048, temperature=0)
    results = batch_inference(inference_dataset, model, sampling_params, 256)
    for i, data in enumerate(ds):
        critic_finetune = results[i]
        data["critic_model"] = "meta-llama/Meta-Llama-3-8B-Finetune"
        data["critic_finetune"] =critic_finetune
        with open(f"{model_name}-gsm8k-test-sdpo_data_critic(base8).json", "a+") as g:
            g.write(json.dumps(data))
            g.write("\n")


def generate_new_test_pipieline():
    ds = []
    model = LLM(model=model_name, tensor_parallel_size=4)
    results = []
    with open(f"{model_name}-gsm8k-test-sdpo_data_critic(base5).json", "r") as f:
        for line in f:
            data = json.loads(line)
            ds.append(data)
    inference_dataset = [
        generate_new_template(item["question"], item["origin_response"], item["critic_finetune"])
        for item in ds
    ]
    sampling_params = SamplingParams(max_tokens=1024, temperature=0)
    results = batch_inference(inference_dataset, model, sampling_params, 256)
    for i, data in enumerate(ds):
        answer = data["answer"]
        new_response_finetune = "Let's break it down step by step:\n\n" + results[i]
        data["new_response_finetune"] = new_response_finetune
        flag1, flag2 = compare_answer(answer, new_response_finetune)
        data["correctness-new-finetune"] = flag1
        with open(f"{model_name}-gsm8k-test-sdpo_data_new(base5).json", "a+") as g:
            g.write(json.dumps(data))
            g.write("\n")
generate_new_test_pipieline()