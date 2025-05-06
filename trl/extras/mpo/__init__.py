import os
from glob import glob
from typing import Any

import jinja2
import regex as re
from natsort import natsorted
from sglang import RuntimeEndpoint

from .mpo_datasets import (
    prepare_essay_writing_dataset,
    prepare_ethical_reasoning_dataset,
    prepare_mathematical_reasoning_dataset,
    prepare_summarization_dataset,
)


def get_task_dataset(task_name: str, tokenizer, split: str):
    """
    Load the task dataset based on the task name.
    """
    assert split in ["train", "test"], f"Split '{split}' is not supported. Please choose from ['train', 'test']"
    from trl.extras import mpo

    if task_name == "essay_writing":
        dataset = prepare_essay_writing_dataset(tokenizer, split)
    elif task_name == "summarization":
        dataset = prepare_summarization_dataset(tokenizer, split)
    elif task_name == "math_reasoning":
        data_file_paths = os.path.join(os.path.dirname(mpo.__file__), "corpora", "MATH", f"{split}/**/*.json")
        dataset = prepare_mathematical_reasoning_dataset(tokenizer, data_file_paths=data_file_paths)
    elif task_name == "ethical_reasoning":
        corpus_filename = "train.scruples-anecdotes.jsonl" if split == "train" else "dev-test.scruples-anecdotes.jsonl"
        data_file_path = os.path.join(os.path.dirname(mpo.__file__), "corpora", "anecdotes", corpus_filename)
        dataset = prepare_ethical_reasoning_dataset(tokenizer, data_file_path=data_file_path)
    else:
        raise ValueError(
            f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        )
    return dataset


def get_reward_model(task_name: str, reward_model_address: str, experiment_directory: str):
    """
    Load the reward model based on the task name.
    """
    from trl.extras.mpo.rm_essay_writing import RewardModelEssayWriting

    # from trl.extras.mpo.rm_summarization import RewardModelSummarization
    # from trl.extras.mpo.rm_math_reasoning import RewardModelMathReasoning
    # from trl.extras.mpo.rm_ethical_reasoning import RewardModelEthicalReasoning

    if task_name == "essay_writing":
        reward_model = RewardModelEssayWriting(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    # elif task_name == "summarization":
    #     reward_model = RewardModelSummarization(
    #         reward_model_address=reward_model_address,
    #         experiment_directory=experiment_directory,
    #     )
    # elif task_name == "math_reasoning":
    #     reward_model = RewardModelMathReasoning(
    #         reward_model_address=reward_model_address,
    #         experiment_directory=experiment_directory,
    #     )
    # elif task_name == "ethical_reasoning":
    #     reward_model = RewardModelEthicalReasoning(
    #         reward_model_address=reward_model_address,
    #         experiment_directory=experiment_directory,
    #     )
    else:
        raise ValueError(
            f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        )
    return reward_model


def get_meta_reward_model(task_name: str, reward_model_address: str, experiment_directory: str):
    """
    Load the meta reward model based on the task name.
    """
    from trl.extras.mpo.rm_essay_writing import MetaRewardModelEssayWriting

    # from trl.extras.mpo.rm_summarization import MetaRewardModelSummarization
    # from trl.extras.mpo.rm_math_reasoning import MetaRewardModelMathReasoning
    # from trl.extras.mpo.rm_ethical_reasoning import MetaRewardModelEthicalReasoning

    if task_name == "essay_writing":
        meta_reward_model = MetaRewardModelEssayWriting(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    # elif task_name == "summarization":
    #     meta_reward_model = MetaRewardModelSummarization(
    #         reward_model_address=reward_model_address,
    #         experiment_directory=experiment_directory,
    #     )
    # elif task_name == "math_reasoning":
    #     meta_reward_model = MetaRewardModelMathReasoning(
    #         reward_model_address=reward_model_address,
    #         experiment_directory=experiment_directory,
    #     )
    # elif task_name == "ethical_reasoning":
    #     meta_reward_model = MetaRewardModelEthicalReasoning(
    #         reward_model_address=reward_model_address,
    #         experiment_directory=experiment_directory,
    #     )
    else:
        raise ValueError(
            f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        )
    return meta_reward_model


class RewardModel:
    """
    Base class for reward models.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        self.reward_model_address = reward_model_address
        self.experiment_directory = experiment_directory
        self.prompts_directory = os.path.join(self.experiment_directory, "prompts")
        self.backend = RuntimeEndpoint(self.reward_model_address)

    def get_latest_rubric_path_and_iteration_index(self) -> str:
        """
        Get the latest evaluation rubric file path along with its iteration index.
        """
        latest_rubric_path = natsorted(glob(os.path.join(self.prompts_directory, "evaluation_rubric_iter_*.txt")))[-1]
        assert os.path.exists(latest_rubric_path), f"Latest rubric path {latest_rubric_path} does not exist!"
        iter_index_m = re.match(r"evaluation\_rubric\_iter\_(\d+)\.txt", os.path.basename(latest_rubric_path))
        assert iter_index_m is not None, f"Failed to parse iteration index from {latest_rubric_path}"
        iter_index = int(iter_index_m.group(1))
        return latest_rubric_path, iter_index

    def read_rubric_items(self, file_path: str) -> list[str]:
        with open(file_path) as f:
            rubric = f.read()
        rgx_rubric_item = r"<item>(.*?)<\/item>"
        item_matches = re.findall(rgx_rubric_item, rubric, re.MULTILINE | re.DOTALL)
        rubric_items = [m.strip() for m in item_matches]
        if len(rubric_items) < 1:
            raise ValueError(f"No rubric items parsed for {file_path}!")
        return rubric_items

    def score(
        self, queries: list[str], responses: list[str], return_evaluations: bool = True, **kwargs
    ) -> list[float]:
        """
        Compute the reward score for a list of queries and responses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the score() method.")


class MetaRewardModel:
    """
    Base class for meta reward models.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        self.reward_model_address = reward_model_address
        self.experiment_directory = experiment_directory
        self.rollouts_directory = os.path.join(self.experiment_directory, "rollouts")
        self.prompts_directory = os.path.join(self.experiment_directory, "prompts")
        self.backend = RuntimeEndpoint(self.reward_model_address)

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.prompts_directory))
        mrm_prescreen_filename = kwargs.get("mrm_prescreen_filename", "mrm_prescreen.txt")
        mrm_analyze_filename = kwargs.get("mrm_analyze_filename", "mrm_analyze.txt")
        mrm_refine_filename = kwargs.get("mrm_refine_filename", "mrm_refine.txt")
        mrm_merge_filename = kwargs.get("mrm_merge_filename", "mrm_merge.txt")
        self.prescreen_template = env.get_template(mrm_prescreen_filename)
        self.analyze_template = env.get_template(mrm_analyze_filename)
        self.refine_template = env.get_template(mrm_refine_filename)
        self.merge_template = env.get_template(mrm_merge_filename)

    def get_latest_rubric_path_and_iteration_index(self) -> str:
        """
        Get the latest evaluation rubric file path along with its iteration index.
        """
        latest_rubric_path = natsorted(glob(os.path.join(self.prompts_directory, "evaluation_rubric_iter_*.txt")))[-1]
        iter_index_m = re.match(r"evaluation\_rubric\_iter\_(\d+).txt", os.path.basename(latest_rubric_path))
        assert iter_index_m is not None, f"Failed to parse iteration index from {latest_rubric_path}"
        iter_index = int(iter_index_m.group(1))
        return latest_rubric_path, iter_index

    def meta_evaluate_and_update(
        self, batch_index: int, return_evaluations: bool = True, num_samples: int = 30, **kwargs
    ) -> dict[str, Any]:
        """
        Meta-evaluate the model, update the evaluation rubric, and return the analyses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the meta_evaluate_and_update() method.")
