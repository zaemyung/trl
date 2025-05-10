import json
import os
import random
from collections import Counter
from glob import glob
from time import time
from typing import Any

import regex as re
from sglang import assistant, function, gen, system, user

from trl.extras.mpo import MetaRewardModel, RewardModel


def parse_task_descriptions_and_prompts(queries: list[str]) -> list[str]:
    rgx_task_and_writing_prompt = r"user(.+?)Bill:\n```(.+?)```"
    task_descriptions = []
    writing_prompts = []
    for q in queries:
        m = re.search(rgx_task_and_writing_prompt, q, re.MULTILINE | re.DOTALL)
        if m is not None:
            task_description = m.group(1).strip()
            writing_prompt = m.group(2).strip()
            task_descriptions.append(task_description)
            writing_prompts.append(writing_prompt)
        else:
            raise ValueError(f"Could not parse task description and writing prompt from: {q}")
    return task_descriptions, writing_prompts


@function
def rm_score(s, task_description: str, writing_prompt: str, response: str, rubric_items: list[str]):
    s += system("You will act as an English instructor.")
    s += user(
        "Given the writing task and specific writing prompt given below, you will assess the quality of a student's essay or story by sequentially assigning a score to each rubric item. "
        + 'For each item, you need to first write a single-sentence rationale followed by a single integer score. Finish your generation with: "<EOE>".\n'
        + "Example of your output should be:\n"
        + "<reason>[Your rationale for assigned score]</reason> <score>[integer score]</score> <EOE>\n"
        + "\n\nTask Instruction:\n“"
        + task_description
        + "”\n\nWriting Prompt:\n“"
        + writing_prompt
        + "”\n\nStudent's Generation:\n“"
        + response
        + "”\n\n"
        + "Example of your output should be:\n"
        + "<reason>[Your rationale for assigned score]</reason> <score>[integer score]</score> <EOE>\n"
        + "\nEnsure that you output only an integer score, enclosed within <score> and </score> tags.\n\n"
    )
    forks = s.fork(len(rubric_items))
    for i, (f, item) in enumerate(zip(forks, rubric_items)):
        f += user(f"\nRubric Item #{i + 1}\n{item}\n\nYour rationale and integer score:\n")
        f += assistant(gen(f"rationale_and_score_{i + 1}", temperature=0.02, max_tokens=400, stop=["<EOE>"]))

    evaluations = []
    for i, f in enumerate(forks):
        try:
            evaluations.append(f[f"rationale_and_score_{i + 1}"])
        except Exception as e:
            print(f"Error processing rubric item {i + 1}: {e}")
            evaluations.append("None")
    s.set_var("evaluations", evaluations)


class RewardModelEthicalReasoning(RewardModel):
    """
    Reward model for essay writing tasks.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address, experiment_directory=experiment_directory, **kwargs
        )
        prompt_path, _ = self.get_latest_rubric_path_and_iteration_index()
        self.rubric_items = self.read_rubric_items(prompt_path)

    def score(
        self, queries: list[str], responses: list[str], return_evaluations: bool = True, *args, **kwargs
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """
        Compute the reward score for a list of queries and responses.
        """
        assert len(queries) == len(responses)
        # queries contain the task description and writing prompt, need to separate them
        task_descriptions, writing_prompts = parse_task_descriptions_and_prompts(queries)
        assert len(task_descriptions) == len(writing_prompts) == len(queries)
        rgx_verdict = r"(<verdict>(RIGHT|WRONG)</verdict>)"
        rgx_anec = r"### Anecdote(.+)### Ethical Analysis and Verdict"

        indices_for_llm = []
        inputs_for_llm = []
        all_states = []
        all_evaluations = []
        for i, (q, r) in enumerate(zip(queries, responses)):
            verdict_part = re.search(rgx_verdict, r, re.MULTILINE | re.DOTALL)
            if verdict_part is None:
                evals = ["<reason> Reasoning is missing verdict.</reason> <score>-10</score>"]
                all_states.append({"evaluations": evals})
                continue

            response_without_verdict = re.sub(rgx_verdict, "", r).strip()
            if len(response_without_verdict.split()) < 30:
                evals = ["<reason> Reasoning is too short.</reason> <score>-10</score>"]
                all_states.append({"evaluations": evals})
                continue

            # otherwise, we need to run the LLM to get the score
            indices_for_llm.append(i)
            try:
                anec = re.search(rgx_anec, q, re.MULTILINE | re.DOTALL).group(1)
            except:
                anec = q
            inputs_for_llm.append({"anec": anec, "response": r, "rubric_items": self.rubric_items})
            all_states.append("dummy state")

        states_from_llm = rm_score.run_batch(inputs_for_llm, backend=self.backend)
        assert len(states_from_llm) == len(inputs_for_llm) == len(indices_for_llm)
        for all_states_index, s in zip(indices_for_llm, states_from_llm):
            assert all_states[all_states_index] == "dummy state"
            try:
                evaluations = s["evaluations"]
            except Exception as e:
                print(f"Could not retrieve s['evaluations'] for state index, {i}: {e}")
                s.set_var(
                    "evaluations",
                    ["<reason> Could not retrieve s['evaluations'] for this sample.</reason> <score>0</score>"],
                )
                all_states[all_states_index] = s
                continue
            all_states[all_states_index] = s

            for eval_index, evaluation in enumerate(s["evaluations"]):
                try:
                    m = re.search(r"<score>(.*?)<\/score>", evaluation, re.MULTILINE | re.DOTALL)
                    if m is not None:
                        score = m.group(1).strip()
                        try:
                            score = float(score)
                        except:
                            try:
                                score = float(eval(score))
                            except:
                                score = 0
                    else:
                        # TODO: ADD TO ALL SCORES
                        print(f"Could not parse <score> </score> from: {evaluation}")
                        s["evaluations"][eval_index] = (
                            "<reason> Could not parse score from `this sample.</reason> <score>0</score>"
                        )
                except Exception as e:
                    print(f"Error processing state index, {i}: {e}")
                    print(f"state['output']: {s['output']}")
                    evaluations[eval_index] = (
                        "<reason> Could not parse score from this sample.</reason> <score>0</score>"
                    )

        if return_evaluations:
            return scores, all_evaluations
        return scores


@function
def mrm_prescreen(s, prescreen_prompt: str):
    s += system("You are a helpful English teacher.")
    s += user(prescreen_prompt)
    s += assistant(gen("verdict", choices=["good", "ok", "bad"]))


@function
def mrm_analyze_and_refine(s, analyze_prompt: str, refine_prompt: str):
    s += system("You are a helpful English teacher.")
    s += user(analyze_prompt)
    s += assistant(gen("analysis", temperature=0.02, max_tokens=2000, stop=["<EOE>"]))
    s += user(refine_prompt)
    s += assistant(gen("refinement", temperature=0.02, max_tokens=2000, stop=["<EOE>", "</rubric>"]))


@function
def mrm_merge(s, merge_prompt: str, temperature: float = 0.02):
    s += system("You are a helpful English teacher.")
    s += user(merge_prompt)
    s += assistant(gen("merged", temperature=temperature, max_tokens=3000, stop=["<EOE>", "</rubric>"]))
    if "<item>" not in s["merged"]:
        s += user(
            "Important: Please rewrite the merged prompt so that each evaluation criterion is clearly enclosed between <item> and </item> tags, exactly as instructed."
        )
        s += assistant(gen("merged", temperature=temperature, max_tokens=3000, stop=["<EOE>", "</rubric>"]))


class MetaRewardModelEthicalReasoning(MetaRewardModel):
    """
    Meta reward model for essay writing tasks.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            **kwargs,
        )

    def join_under_word_limit(self, refinements: list[str], N: int, sep: str = "\n===\n") -> str:
        """
        Join strings from `refinements` with `sep`, stopping before the total
        whitespace‑separated word count exceeds `N`.

        Parameters
        ----------
        refinements : list[str]
            The strings to join.
        N : int
            Maximum total number of words allowed in the joined result.
        sep : str, optional
            Separator to place between pieces. Defaults to "\n===\n".

        Returns
        -------
        str
            The joined string whose word count is ≤ N.
        """
        joined_parts = []
        word_count = 0

        for chunk in refinements:
            words_in_chunk = len(chunk.split())
            if word_count + words_in_chunk > N:
                break
            joined_parts.append(chunk)
            word_count += words_in_chunk

        return sep.join(joined_parts)

    def meta_evaluate_and_update(
        self,
        batch_index: int,
        return_evaluations: bool = True,
        num_samples: int = 20,
        do_prescreening: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Meta-evaluate the model, update the evaluation rubric, and return the analyses.
        """
        current_junior_prompt_path, current_iter_no = self.get_latest_rubric_path_and_iteration_index()
        with open(current_junior_prompt_path) as f:
            current_junior_prompt = f.read().strip()

        rollouts_paths = glob(os.path.join(self.rollouts_directory, f"rollouts-{batch_index}-*.json"))

        inputs = []
        prescreen_inputs = []
        for path in rollouts_paths:
            with open(path) as f:
                rollouts = json.load(f)
            queries = [sample["queries"] for sample in rollouts]
            _task_descriptions, _writing_prompts = parse_task_descriptions_and_prompts(queries)

            for i, sample in enumerate(rollouts):
                _input = {
                    "task_description": _task_descriptions[i],
                    "student_prompt": _writing_prompts[i],
                    "student_generation": sample["student_responses"],
                    "junior_prompt": current_junior_prompt,
                    "junior_score": sample["junior_scores"],
                }
                inputs.append(_input)
                prescreen_inputs.append(self.prescreen_template.render(_input))

        if do_prescreening:
            ### Prescreening
            print(f"Prescreening {len(prescreen_inputs)} samples...")
            start_time = time()
            states = mrm_prescreen.run_batch(
                [{"prescreen_prompt": _input} for _input in prescreen_inputs], backend=self.backend
            )
            assert len(states) == len(inputs) == len(prescreen_inputs)
            prescreened_verdicts = []
            prescreened_ok_indices = []
            prescreened_bad_indices = []
            for i, s in enumerate(states):
                try:
                    s["verdict"] = s["verdict"].strip()
                except Exception as e:
                    print(f"Could not retrieve s['verdict'] for state index: {e}")
                    s.set_var("verdict", "None")
                prescreened_verdicts.append(s["verdict"])
                if s["verdict"] == "ok":
                    prescreened_ok_indices.append(i)
                elif s["verdict"] == "bad":
                    prescreened_bad_indices.append(i)
            prescreened_verdicts_counter = Counter(prescreened_verdicts)

            end_time = time()
            prescreen_time = (end_time - start_time) / 60
            print(f"Prescreening took {prescreen_time:.2f} minutes for {len(prescreen_inputs)} samples.")
            print(f"Prescreened verdicts dist.: {prescreened_verdicts_counter}")

            random.shuffle(prescreened_bad_indices)
            random.shuffle(prescreened_ok_indices)
            selected_indices = prescreened_bad_indices[:num_samples]
            if len(selected_indices) < num_samples:
                selected_indices += prescreened_ok_indices[: num_samples - len(selected_indices)]
        else:
            selected_indices = random.sample(range(len(inputs)), num_samples)

        ### Analysis and Refinement
        print("Analysis and Refinment step...")
        start_time = time()
        analyze_prompt_all = []
        refine_prompt_all = []
        for index in selected_indices:
            sample = {"max_words": 1000, **inputs[index]}
            analyze_prompt = self.analyze_template.render(sample)
            analyze_prompt_all.append(analyze_prompt)

            refine_prompt = self.refine_template.render({"max_words": 1000})
            refine_prompt_all.append(refine_prompt)

        analysis_and_refinement_inputs = [
            {"analyze_prompt": a_p, "refine_prompt": r_p} for a_p, r_p in zip(analyze_prompt_all, refine_prompt_all)
        ]
        states = mrm_analyze_and_refine.run_batch(analysis_and_refinement_inputs, backend=self.backend)
        analyses = []
        refinements = []
        for i, s in enumerate(states):
            analysis = s["analysis"]
            analyses.append(analysis)
            refinement = s["refinement"]
            refinements.append(f"Junior Instructor's Scoring Criteria Set #{i + 1}:\n{refinement}\n")
        assert len(analyses) == len(refinements) == len(selected_indices)
        end_time = time()
        analysis_time = (end_time - start_time) / 60
        print(f"Analysis and Refinement took {analysis_time:.2f} minutes for {len(selected_indices)} samples.")

        ### Merge the refinements
        print("Merge step...")
        start_time = time()
        joined_prompts = self.join_under_word_limit(refinements, int(32768 / 1.7), sep="\n===\n")
        merge_prompt = self.merge_template.render({"multiple_sets": joined_prompts + "\n```", "max_words": 1200})
        state = mrm_merge.run(merge_prompt=merge_prompt, backend=self.backend)
        merged_criteria = state["merged"]

        regex = r"<item>(.*?)<\/item>"
        matches = re.findall(regex, merged_criteria, re.MULTILINE | re.DOTALL)
        rubric_items = [m.strip() for m in matches]
        if len(rubric_items) < 1 or len(merged_criteria) < 400:
            print("merge step failed! Trying again...")
            state = mrm_merge.run(merge_prompt=merge_prompt, temperature=0.5, backend=self.backend)
            merged_criteria = state["merged"]

        next_junior_prompt_path = os.path.join(
            self.prompts_directory, f"evaluation_rubric_iter_{current_iter_no + 1}.txt"
        )
        with open(next_junior_prompt_path, "w") as f:
            f.write(f"{merged_criteria.strip()}\n")
        end_time = time()
        merge_time = (end_time - start_time) / 60
        print(f"Merge took {merge_time:.2f} minutes. New version saved to {next_junior_prompt_path}")

        if return_evaluations:
            assert len(analyses) == len(refinements) == len(selected_indices)
            logs = []
            for i, selected_index in enumerate(selected_indices):
                log = {
                    "junior_prompt": current_junior_prompt,
                    "meta_analysis": analyses[i],
                    "meta_refinement": refinements[i],
                    **inputs[selected_index],
                }
                if do_prescreening:
                    log["prescreened_verdict"] = prescreened_verdicts[selected_index]
                logs.append(log)
            return logs
