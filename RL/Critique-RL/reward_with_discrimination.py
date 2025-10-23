import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from data.build_data import(
    ChatMLTemplate,
    Llama2Template,
    Llama3Template
)
from transformers import AutoTokenizer
from typing import Optional, List
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from openrlhf.utils.logging_utils import init_logger
import logging
from reward.reward_func import RewardFunc, RefineRewardFunc, DiscriminationRewardFunc


def get_logger(log_path: str):
    logger = init_logger(__name__)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    logger.addHandler(handler)
    return logger


logger = get_logger("./ppo_generate_log.log")


@dataclass
class ScriptAuguments:
    actor_model: Optional[str]=field(default='Qwen/Qwen2.5-0.5B', metadata={"help": "the actor model path"})
    critique_model: Optional[str]=field(default='Qwen/Qwen2.5-0.5B', metadata={"help": "the critique model path"})
    template: Optional[str]=field(default='chatml', metadata={"help": "the template name"})
    dataset_type: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "dataset types (accept multiple values, e.g. math gsm8k aqua)",
            "nargs": "+"  # or '*'
        }
    )
    tensor_parallel_size: Optional[int] = field(default=1, metadata={"help": "tensor_parallel_size for vllm"})
    gpu_memory_utilization: Optional[float] = field(default=0.5, metadata={"help": "gpu_memory_utilization for vllm"})
    temperature: Optional[float] = field(default=0, metadata={"help": "the sampling temperature"})
    port: Optional[int]=field(default=5000, metadata={"help": "the port where the reward server runs"})
    rf_mode: Optional[str]=field(default='r_refine', metadata={"help": "reward function"})
    use_discrimination: Optional[bool]=field(default=False, metadata={"help": "whether use discrimination reward function"})
    discrimination_only_step: Optional[int]=field(default=200, metadata={"help": "steps to train discrimination ability"})
    host: Optional[str]=field(default='0.0.0.0', metadata={"help": "the host where the reward server runs"})



class RewardModelProxy:
    def __init__(self, args: ScriptAuguments):
        self.actor_model = args.actor_model
        assert args.template in ["chatml", "llama3", "llama2"]
        template_dict = {
            "chatml": ChatMLTemplate(),
            "llama2": Llama2Template(),
            "llama3": Llama3Template()
        }
        self.dataset_types = args.dataset_type 
        self.template = template_dict[args.template]
        self.tokenizer = AutoTokenizer.from_pretrained(args.critique_model)
        self.discrimination_reward_func = DiscriminationRewardFunc(self.template, discimination_correct_score=1.0)
        self.refinement_reward_func = None
        self.discrimination_only_step = args.discrimination_only_step
        self.tensor_parallel_size = args.tensor_parallel_size
        self.gpu_memory_utilization = args.gpu_memory_utilization
        self.temperature = args.temperature
        self.rf_mode = args.rf_mode
        self.use_discrimination = args.use_discrimination
        
        

    def get_clean_critique(self, raw_critiques):
        clean_critiques=[]
        for idx, critique in enumerate(raw_critiques):
            # get critique response
            ast_begin = self.template.ast_begin_mark
            end_mark = self.template.conv_end_mark
            critique = critique.split(ast_begin)[-1].strip().replace(end_mark, "")
            # delete other special tokens(padding)
            special_tokens = [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]
            for token in special_tokens:
                if token is not None:
                    critique = critique.replace(token, "")
            clean_critiques.append(critique)
        return clean_critiques
        
    def get_reward(self, critiques, rawdatas, step):
        # batch
        with torch.no_grad():
            clean_critiques = self.get_clean_critique(critiques)
            # get discrimination reward
            if step <= self.discrimination_only_step:
                return self.discrimination_reward_func.get_reward(self.dataset_types, clean_critiques, rawdatas)
            else:
                # get refine disc reward
                if self.refinement_reward_func is None:
                    self.refinement_reward_func = RefineRewardFunc(
                        actor_model=self.actor_model,
                        template=self.template,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        temperature=self.temperature,
                        rf_mode=self.rf_mode,
                        use_discrimination=self.use_discrimination
                    )
                return self.refinement_reward_func.get_reward(self.dataset_types, clean_critiques, rawdatas)

if __name__ == '__main__':
    parser = HfArgumentParser(ScriptAuguments)
    args = parser.parse_args_into_dataclasses()[0]
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        critiques = data.get("critiques")
        rawdatas = data.get("rawdatas")
        step = data.get("step")
        all_scores = reward_model.get_reward(critiques, rawdatas, step)
        
        result = {"rewards": all_scores}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
