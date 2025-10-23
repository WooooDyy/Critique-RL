from transformers import PreTrainedTokenizerBase
from typing import List
from abc import ABC, abstractmethod

class BaseChatTemplate(ABC):
    # prompts
    inference_user_prompt="{question} Please also give your final number in the format of 'The answer is [your answer].'"
    inference_agent_prompt="Let's break it down step by step:\n\n"

    critique_user_prompt="""Instruction:
For a given question, there is a step-by-step solution of which each sentence is a step. The solution may contain some errors, your task is to check the solution step by step carefully then tell me whether the solution is correct or wrong. Furthermore, you should repeat each step, state the correctness of the step, and give a detailed explanation of why this step is correct or wrong. Finally, you should tell me whether the final answer is correct or wrong.

Your answer format should be:

Step sentence: [Repeated sentence, ignoring new line break]
Correctness of the step: [Correct/Wrong]
Explanation: [Your Explanation]

Correctness of the final answer: [Correct/Wrong]
Explanation: [Your Explanation]


Question:
{question}

Solution:
{solution}"""

    refine_user_prompt="I have made a step-by-step critique for your solution. Please read and understand the critique carefully, paying attention to each step\u2019s correctness and the explanation provided. Then strictly adhere to the critique's guidance to provide a new detailed, step-by-step solution, including all reasoning and intermediate steps. Ensure the new solution aligns with the critique's guidance. Please also give your final number in the format of 'The answer is [your answer].'\n\n{critique}"
    refine_agent_prompt="Let's break it down step by step, strictly following the critique:\n\n"

    ast_begin_mark=""
    conv_end_mark=""

    # template functions
    @abstractmethod
    def template_inference(self, 
                           question: str):
        raise NotImplementedError
    
    @abstractmethod
    def template_critique(self, 
                          question: str, 
                          solution: str):
        raise NotImplementedError
    
    @abstractmethod
    def template_refinement(self, 
                            question: str, 
                            solution: str, 
                            critique: str):
        raise NotImplementedError


class DefaultTemplate(BaseChatTemplate):
    def __init__(self, endoftext):
        self.ast_begin_mark=""
        self.endoftext = endoftext
        self.conv_end_mark= endoftext
   
    def template_inference(self,
                           question: str):
        text=f"Human: {self.inference_user_prompt.format(question=question)}\n\nAssistant:{self.inference_agent_prompt}"
        return text
   
    def template_critique(self,
                          question: str,
                          solution: str):
        if "Let's break it down step by step:\n\n" in solution:
            solution = solution[len("Let's break it down step by step:\n\n"):]
        text = f"Human: {self.critique_user_prompt.format(question=question, solution=solution)}\n\nAssistant:"
        return text
   

    def template_refinement(self,
                            question: str,
                            solution: str,
                            critique: str):
        idx = critique.find("Step sentence:") # ignore words before the first step sentence
        if idx != -1:
            critique = critique[idx:]
        text = f"Human: {self.inference_user_prompt.format(question=question)}\n\nAssistant:{solution}\n{self.endoftext}\nHuman: {self.refine_user_prompt.format(critique=critique)}\n\nAssistant:{self.refine_agent_prompt}"
        return text

class ChatMLTemplate(BaseChatTemplate):
    def __init__(self):
        self.ast_begin_mark="<|im_start|>assistant"
        self.conv_end_mark="<|im_end|>"

    def template_inference(self, 
                           question: str):
        text=f"<|im_start|>user\n{self.inference_user_prompt.format(question=question)}<|im_end|>\n<|im_start|>assistant\n{self.inference_agent_prompt}"
        return text
    
    def template_critique(self, 
                          question: str, 
                          solution: str):
        if "Let's break it down step by step:\n\n" in solution:
            solution = solution[len("Let's break it down step by step:\n\n"):]
        text = f"<|im_start|>user\n{self.critique_user_prompt.format(question=question, solution=solution)}<|im_end|>\n<|im_start|>assistant\n"
        return text

    def template_refinement(self,
                            question: str,
                            solution: str,
                            critique: str):
        idx = critique.find("Step sentence:") # ignore words before the first step sentence
        if idx != -1:
            critique = critique[idx:]
        text = f"<|im_start|>user\n{self.inference_user_prompt.format(question=question)}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>\n<|im_start|>user\n{self.refine_user_prompt.format(critique=critique)}<|im_end|>\n<|im_start|>assistant\n{self.refine_agent_prompt}"
        return text


class Llama2Template(BaseChatTemplate):
    def __init__(self):
        self.ast_begin_mark = "[/INST]"
        self.conv_end_mark = "</s>"
    
    def template_inference(self, 
                           question: str):
        text=f"<s>[INST] {self.inference_user_prompt.format(question=question)} [/INST] {self.inference_agent_prompt}"
        return text
    
    def template_critique(self, 
                          question: str, 
                          solution: str):
        if "Let's break it down step by step:\n\n" in solution:
            solution = solution[len("Let's break it down step by step:\n\n"):]
        text = f"<s>[INST] {self.critique_user_prompt.format(question=question, solution=solution)} [/INST] "
        return text
    
    def template_refinement(self,
                            question: str,
                            solution: str,
                            critique: str):
        idx = critique.find("Step sentence:") # ignore words before the first step sentence
        if idx != -1:
            critique = critique[idx:]
        text = f"<s>[INST] {self.inference_user_prompt.format(question=question)} [/INST] {solution}</s><s>[INST] {self.refine_user_prompt.format(critique=critique)} [/INST] {self.refine_agent_prompt}"
        return text
    
class Llama3Template(BaseChatTemplate):
    def __init__(self):
        self.ast_begin_mark="<|start_header_id|>assistant<|end_header_id|>"
        self.conv_end_mark="<|eot_id|>"

    def template_inference(self, 
                           question: str):
        text=f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.inference_user_prompt.format(question=question)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{self.inference_agent_prompt}"
        return text
    
    def template_critique(self, 
                          question: str, 
                          solution: str):
        if "Let's break it down step by step:\n\n" in solution:
            solution = solution[len("Let's break it down step by step:\n\n"):]
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.critique_user_prompt.format(question=question, solution=solution)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return text

    def template_refinement(self,
                            question: str,
                            solution: str,
                            critique: str):
        idx = critique.find("Step sentence:") # ignore words before the first step sentence
        if idx != -1:
            critique = critique[idx:]
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.inference_user_prompt.format(question=question)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{solution}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{self.refine_user_prompt.format(critique=critique)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{self.refine_agent_prompt}"
        return text

def tokenize_batch(query: List[str], tokenizer: PreTrainedTokenizerBase):
    return tokenizer(query, padding=True, return_tensors='pt')

def template_data(data, template: BaseChatTemplate, task_type: str):
    assert task_type in ['inference', 'critique', 'refinement']
    def template_task(sample):
        if task_type == 'inference':
            return template.template_inference(question=sample['question'])
        elif task_type == 'critique':
            return template.template_critique(question=sample['question'], solution=sample['origin_response'])
        elif task_type == 'refinement':
            return template.template_refinement(question=sample['question'], solution=sample['origin_response'], critique=sample['critique'])

    new_data = template_task(data)
    return new_data

def template_dataset(batch_data, template: BaseChatTemplate, task_type: str):
    return [template_data(data, template, task_type) for data in batch_data]

def build_dataset(data, template, tokenizer, task_type):
    # add template
    prompts = template_dataset(batch_data=data, template=template, task_type=task_type)
    # tokenize
    tokenized = tokenize_batch(prompts, tokenizer=tokenizer)
    batch_data = {key: [d[key] for d in data] for key in data[0].keys()}
    batch_data.update(tokenized)
    return batch_data
