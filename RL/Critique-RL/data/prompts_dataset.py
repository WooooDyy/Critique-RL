from torch.utils.data import Dataset
from tqdm import tqdm
from .build_data import(
    ChatMLTemplate,
    Llama2Template,
    Llama3Template,
    template_data
)

class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        template: chat template
        task: current task(critique, refinement, inference)
    """

    def __init__(
        self,
        dataset,
        strategy,
        task_type
    ) -> None:
        super().__init__()
        self.strategy = strategy
        assert self.strategy.args.critique_template in ["chatml","llama2","llama3"]
        template_dict = {
            "chatml": ChatMLTemplate(),
            "llama2": Llama2Template(),
            "llama3": Llama3Template()
        }

        # chat_template
        self.template = template_dict[self.strategy.args.critique_template]

        self.prompts = []
        self.raw_data = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = template_data(data, self.template, task_type)
            self.prompts.append(prompt)
            self.raw_data.append(data)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.raw_data[idx]
