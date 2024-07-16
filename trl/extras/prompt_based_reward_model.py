from typing import List
import numpy as np
from pydantic import BaseModel, Field, computed_field, ValidationError
import pydantic_core
from sglang import (
    function,
    gen,
    set_default_backend,
    RuntimeEndpoint,
)


class SyntacticScore(BaseModel):
    grammar: float = Field(default=0)
    structure: float = Field(default=0)

    @computed_field(return_type=float)
    @property
    def average(self):
        return np.mean([self.grammar, self.structure])


class SemanticScore(BaseModel):
    clarity: float = Field(default=0)
    relevance: float = Field(default=0)
    vocabulary: float = Field(default=0)

    @computed_field(return_type=float)
    @property
    def average(self):
        return np.mean([self.clarity, self.relevance, self.vocabulary])


class DiscourseScore(BaseModel):
    flow: float = Field(default=0)
    organization: float = Field(default=0)
    balance: float = Field(default=0)

    @computed_field(return_type=float)
    @property
    def average(self):
        return np.mean([self.flow, self.organization, self.balance])


class Evaluation(BaseModel):
    syntactic: SyntacticScore
    semantic: SemanticScore
    discourse: DiscourseScore

    @computed_field(return_type=float)
    @property
    def average(self):
        return np.mean(
            [self.syntactic.average, self.semantic.average, self.discourse.average]
        )


@function
def _score(s, query: str, response: str, overall_guideline: str):
    s += (
        overall_guideline
        + "\n\nInstruction:\n“"
        + query
        + "”\n\nResponse:\n“"
        + response
        + "”\n\nEvaluation JSON Result:\n"
    )
    score_regex = (
        r"""\{\n"""
        + r"""    "syntactic": \{\n"""
        + r"""        "grammar": [012345]{1},\n"""
        + r"""        "structure": [012345]{1}\n"""
        + r"""    \},\n"""
        + r"""    "semantic": \{\n"""
        + r"""        "clarity": [012345]{1},\n"""
        + r"""        "relevance": [012345]{1},\n"""
        + r"""        "vocabulary": [012345]{1}\n"""
        + r"""    \},\n"""
        + r"""    "discourse": \{\n"""
        + r"""        "flow": [012345]{1},\n"""
        + r"""        "organization": [012345]{1},\n"""
        + r"""        "balance": [012345]{1}\n"""
        + r"""    \}\n"""
        + r"""\}"""
    )
    s += gen("output", temperature=0, max_tokens=100, regex=score_regex)


@function
def _correct(s, query: str, prompt: str):
    s += (
        prompt
        + "\n### Original JSON\n“"
        + query
        + "”\n\nOutput <EOE> at the end of refinement.\n\n### Refined JSON\n"
    )
    s += gen("output", temperature=0, max_tokens=200, stop="<EOE>")


class Scorer:
    def __init__(
        self,
        instruction_prompt_path: str = "/home/ubuntu/Development/trl/trl/extras/prompt_v3.txt",
        correction_prompt_path: str = "/home/ubuntu/Development/trl/trl/extras/json_correction_prompt.txt",
        reward_model_address: str = "http://172.31.36.14:8501",
    ) -> None:
        set_default_backend(RuntimeEndpoint(reward_model_address))
        with open(instruction_prompt_path, "r") as inf:
            self.instruction_prompt = inf.read()
        with open(correction_prompt_path, "r") as inf:
            self.correction_prompt = inf.read()

    def score(self, queries: List[str], responses: List[str]) -> List[float]:
        assert len(queries) == len(responses)
        inputs = [
            {"query": q, "response": r, "overall_guideline": self.instruction_prompt}
            for q, r in zip(queries, responses)
        ]
        states = _score.run_batch(inputs)
        scores = []
        for s in states:
            try:
                _json_output = pydantic_core.from_json(s["output"], allow_partial=True)
                score = Evaluation.model_validate(_json_output)
                scores.append(score)
            except Exception as e:
                print(e)
                print(s["output"])
                result = _correct.run(query=s["output"], prompt=self.correction_prompt)
                corr_score = Evaluation.model_validate(
                    pydantic_core.from_json(result["output"], allow_partial=True)
                )
                scores.append(corr_score)
        return [s.average for s in scores]


if __name__ == "__main__":
    responses = [
        "   ",
        "<|endoftext|>",
        "Me.",
        "The night is young, and by the grace of magic, so are we. The phrase conveys a sense of a magical night full of possibilities, where the enchanting atmosphere makes us feel young and revitalized, ready to embrace whatever the night has to offer.",
        "It's me.<|endoftext|>",
        "Dear editor<|endoftext|>",
        "Pam is pam. Me is me.",
        "The moment he realized it, it was not so good. But it was ok.",
        "The moment he realized it, it was not so good. But it was ok.",
    ]
    queries = ["Generate an essay."] * len(responses)
    ds = Scorer()
    scores = ds.score(queries=queries, responses=responses)

    for s in scores:
        print(s)
        print("-------------")
