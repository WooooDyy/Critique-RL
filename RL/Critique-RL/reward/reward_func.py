import torch
from vllm import LLM, SamplingParams
from data.build_data import(
    BaseChatTemplate,
    template_dataset
)
from abc import ABC, abstractmethod
from process_ans.utils import compare_answer


class RewardFunc(ABC):
    def __init__(self, template: BaseChatTemplate) -> None:
        self.template = template

    @abstractmethod
    def get_reward(self, dataset_types, critiques, rawdatas):
        raise NotImplementedError


class DiscriminationRewardFunc(RewardFunc):
    def __init__(self, template: BaseChatTemplate, discimination_correct_score: float=1.0) -> None:
        super().__init__(template)
        self.discimination_correct_score = discimination_correct_score
    
    def get_reward(self,
                   dataset_types,
                   critiques,
                   rawdatas
        ):
        with torch.no_grad():
            correct_critique_state = "Correctness of the final answer: Correct"
            wrong_critique_state = "Correctness of the final answer: Wrong"
            discriminations = []
            discrimination_acc = []
            origin_scores = []
            scores = []
            correctness_origins = rawdatas["correctness-origin"]
            rawdatas["critique"]=critiques
            for critique, correctness_origin in zip(critiques, correctness_origins):
                origin_scores.append(correctness_origin)
                if correctness_origin:
                    if correct_critique_state in critique:
                        discriminations.append(1.0)
                        discrimination_acc.append(1.0)
                        scores.append(self.discimination_correct_score)
                    elif wrong_critique_state in critique:
                        discriminations.append(-1.0)
                        discrimination_acc.append(0.0)
                        scores.append(-1.0)
                    else:
                        discriminations.append(-1.0)
                        discrimination_acc.append(0.0)
                        scores.append(-1.0)
                else:
                    if wrong_critique_state in critique:
                        discriminations.append(1.0)
                        discrimination_acc.append(1.0)
                        scores.append(self.discimination_correct_score)
                    elif correct_critique_state in critique:
                        discriminations.append(-1.0)
                        discrimination_acc.append(0.0)
                        scores.append(-1.0)
                    else:
                        discriminations.append(-1.0)
                        discrimination_acc.append(0.0)
                        scores.append(-1.0)
            reward_results = {
                "discriminations": discriminations,
                "discrimination_acc": discrimination_acc,
                "origin_scores": origin_scores,
                "scores": scores
            }
            for dataset_type in dataset_types:
                reward_results[f"is_{dataset_type}"] = [0.0 for _ in range(len(rawdatas["dataset_type"]))]
            for idx, dataset_type in enumerate(rawdatas["dataset_type"]):
                reward_results[f"is_{dataset_type}"][idx] = 1.0
            return reward_results
    
class RefineRewardFunc(RewardFunc):
    def __init__(self, 
                 actor_model: str,
                 template: BaseChatTemplate,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.95,
                 temperature: float = 0.5,
                 rf_mode: str="rf_mode4",
                 use_discrimination: bool=True,
                 ) -> None:
        super().__init__(template)
        self.actor_model = actor_model
        self.llm = LLM(model=actor_model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
        stop_tokens = ["<|end_of_text|>", "</s>", "Human", "Assistant", "<|im_end|>","<|endoftext|>"]
        self.sampling_params = SamplingParams(max_tokens=1024, temperature=temperature, stop=stop_tokens)
        self.rf_mode = rf_mode
        self.use_discrimination = use_discrimination
        self.discrimination_reward_func = DiscriminationRewardFunc(template, discimination_correct_score=0.2)
    
    def reward_score_func(self, origin_correctness, refinement_correctness, rf_mode="r_refine"):
        assert rf_mode in ["r_refine", "r_correct", "r_delta"]
        if rf_mode == "r_correct":
            if origin_correctness == 1.0 and refinement_correctness == 1.0:
                return 0.2
            if origin_correctness == 1.0 and refinement_correctness == 0.0:
                return 0.0
            if origin_correctness == 0.0 and refinement_correctness == 1.0:
                return 1.0
            if origin_correctness == 0.0 and refinement_correctness == 0.0:
                return 0.0
        elif rf_mode == "r_delta":
            if origin_correctness == 1.0 and refinement_correctness == 1.0:
                return 0.0
            if origin_correctness == 1.0 and refinement_correctness == 0.0:
                return -1.0
            if origin_correctness == 0.0 and refinement_correctness == 1.0:
                return 1.0
            if origin_correctness == 0.0 and refinement_correctness == 0.0:
                return 0.0
        elif rf_mode == "r_refine":
            return 1.0 if refinement_correctness==1.0 else 0.0

    def get_reward(self, dataset_types, critiques, rawdatas):
        origin_scores = []
        refinement_scores = []
        scores = []
        correct_to_correct = []
        correct_to_incorrect = []
        incorrect_to_correct = []
        incorrect_to_incorrect = []
        delta = []
        change_rate = []

        # batch
        with torch.no_grad():
            rawdatas["critique"]=critiques
            batch_data = [
            {key: value[i] for key, value in rawdatas.items()}
            for i in range(len(next(iter(rawdatas.values()))))
            ]
            prompts = template_dataset(batch_data=batch_data, template=self.template, task_type="refinement")
            outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            for idx, response in enumerate(responses):
                answer = rawdatas["answer"][idx]
                origin_response = rawdatas["origin_response"][idx]
                dataset_type = rawdatas["dataset_type"][idx]
                refinement_correctness = compare_answer(answer, response, dataset_type) 
                has_change = not compare_answer(origin_response, response, dataset_type)
                origin_correctness = rawdatas["correctness-origin"][idx]
                score = self.reward_score_func(origin_correctness, refinement_correctness, rf_mode=self.rf_mode)
                scores.append(score)
                origin_scores.append(origin_correctness)
                refinement_scores.append(refinement_correctness)
                delta.append(refinement_correctness - origin_correctness)
                change_rate.append(has_change)
                
                if origin_correctness == 1.0 and  refinement_correctness == 1.0:
                    correct_to_correct.append(1.0)
                    correct_to_incorrect.append(0.0)
                    incorrect_to_correct.append(0.0)
                    incorrect_to_incorrect.append(0.0)
                elif origin_correctness == 1.0 and  refinement_correctness == 0.0:
                    correct_to_correct.append(0.0)
                    correct_to_incorrect.append(1.0)
                    incorrect_to_correct.append(0.0)
                    incorrect_to_incorrect.append(0.0)
                elif origin_correctness == 0.0 and  refinement_correctness == 1.0:
                    correct_to_correct.append(0.0)
                    correct_to_incorrect.append(0.0)
                    incorrect_to_correct.append(1.0)
                    incorrect_to_incorrect.append(0.0)
                elif origin_correctness == 0.0 and  refinement_correctness == 0.0:
                    correct_to_correct.append(0.0)
                    correct_to_incorrect.append(0.0)
                    incorrect_to_correct.append(0.0)
                    incorrect_to_incorrect.append(1.0)
            # get discrimination reward
            discrimination_reward = self.discrimination_reward_func.get_reward(dataset_types, critiques, rawdatas)
            # update scores
            if self.use_discrimination:
                scores = [s + d if d != -1.0 else d for s, d in zip(scores, discrimination_reward["scores"])]
            reward_results = {
                "origin_scores": origin_scores,
                "refinement_scores": refinement_scores,
                "delta": delta,
                "scores": scores,
                "correct_to_correct": correct_to_correct,
                "correct_to_incorrect": correct_to_incorrect,
                "incorrect_to_correct": incorrect_to_correct,
                "incorrect_to_incorrect": incorrect_to_incorrect,
                "change_rate": change_rate
            }
            # add discrimination
            discrimination_reward.pop("scores")
            discrimination_reward.pop("origin_scores")
            reward_results.update(discrimination_reward)
            for dataset_type in dataset_types:
                reward_results[f"is_{dataset_type}"] = [0.0 for _ in range(len(rawdatas["dataset_type"]))]
            for idx, dataset_type in enumerate(rawdatas["dataset_type"]):
                reward_results[f"is_{dataset_type}"][idx] = 1.0
            return reward_results
