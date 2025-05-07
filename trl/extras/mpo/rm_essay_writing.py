import json
import os
import random
from glob import glob
from time import time
from typing import Any

import numpy as np
import regex as re
from sglang import assistant, function, gen, system, user

from trl.extras.mpo import MetaRewardModel, RewardModel


def parse_task_descriptions_and_prompts(queries: list[str]) -> list[str]:
    rgx_task_and_writing_prompt = r"user(.+?)Instructions:(.+?)Your Writing:"
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


class RewardModelEssayWriting(RewardModel):
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

        states = []
        inputs = [
            {
                "task_description": t,
                "writing_prompt": w_p,
                "response": r,
                "rubric_items": self.rubric_items,
            }
            for t, w_p, r in zip(task_descriptions, writing_prompts, responses)
        ]
        states = rm_score.run_batch(inputs, backend=self.backend)
        scores = []
        all_evaluations = []
        for i, s in enumerate(states):
            # s contains results for each query and response pair
            try:
                evaluations = s["evaluations"]
            except Exception as e:
                print(f"Could not retrieve s['evaluations'] for state index, {i}: {e}")
                scores.append(0)
                s.set_var("evaluations", None)
                all_evaluations.append(None)
                continue
            sub_scores = []
            for evaluation in evaluations:
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
                        print(f"Could not parse <score> </score> from: {evaluation}")
                        score = 0
                except Exception as e:
                    print(f"Error processing state index, {i}: {e}")
                    print(f"state['output']: {s['output']}")
                    score = 0
                sub_scores.append(score)
            scores.append(sum(sub_scores))
            all_evaluations.append(evaluations)
        if return_evaluations:
            return scores, all_evaluations
        return scores


@function
def mrm_prescreen(s, prescreen_prompt: str):
    s += system("You are a helpful English teacher.")
    s += user(prescreen_prompt)
    s += assistant(gen("score", choices=["0", "1"]))


@function
def mrm_analyze_and_refine(s, analyze_prompt: str, refine_prompt: str):
    s += system("You are a helpful English teacher.")
    s += user(analyze_prompt)
    s += assistant(gen("analysis", temperature=0.02, max_tokens=3000, stop=["<EOE>"]))
    s += user(refine_prompt)
    s += assistant(gen("refinement", temperature=0.02, max_tokens=3000, stop=["<EOE>", "</rubric>"]))


@function
def mrm_merge(s, merge_prompt: str, temperature: float = 0.02):
    s += system("You are a helpful English teacher.")
    s += user(merge_prompt)
    s += assistant(gen("merged", temperature=temperature, max_tokens=3000, stop=["<EOE>", "</rubric>"]))


class MetaRewardModelEssayWriting(MetaRewardModel):
    """
    Meta reward model for essay writing tasks.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            **kwargs,
        )

    def meta_evaluate_and_update(
        self, batch_index: int, return_evaluations: bool = True, num_samples: int = 20, **kwargs
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

        ### Prescreening
        print(f"Prescreening {len(prescreen_inputs)} samples...")
        start_time = time()
        states = mrm_prescreen.run_batch(
            [{"prescreen_prompt": _input} for _input in prescreen_inputs], backend=self.backend
        )
        assert len(states) == len(inputs) == len(prescreen_inputs)
        prescreened_indices = []
        prescreened_scores = []
        for i, s in enumerate(states):
            try:
                s["score"] = int(s["score"])
            except Exception as e:
                print(f"Could not retrieve s['score'] for state index: {e}")
                s.set_var("score", None)
                continue
            if s["score"] == 0:
                prescreened_indices.append(i)
            prescreened_scores.append(s["score"])

        end_time = time()
        prescreen_time = (end_time - start_time) / 60
        print(f"Prescreening took {prescreen_time:.2f} minutes for {len(prescreen_inputs)} samples.")
        print(
            f"Prescreened scores -> 1s: {np.sum(prescreened_scores)}, 0s: {len(prescreened_scores) - np.sum(prescreened_scores)}"
        )

        random.shuffle(prescreened_indices)
        selected_indices = prescreened_indices[:num_samples]

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
        merge_prompt = self.merge_template.render(
            {"multiple_sets": "\n===\n".join(refinements) + "\n```", "max_words": 1000}
        )
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
                logs.append(
                    {
                        "junior_prompt": current_junior_prompt,
                        "meta_analysis": analyses[i],
                        "meta_refinement": refinements[i],
                        **inputs[selected_index],
                    }
                )
            return logs


if __name__ == "__main__":
    reward_model_address = "http://129.213.31.51:30000"
    experiment_directory = "/home/ubuntu/temp/mytest"
    prompts_directory = os.path.join(experiment_directory, "prompts")
    batch_index = 10

    reward_model = RewardModelEssayWriting(
        reward_model_address=reward_model_address,
        experiment_directory=experiment_directory,
    )

    queries = [
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues against the claim that the majority of insurance types should be managed by non-profit organizations or the government. In your essay, provide evidence to support your stance, including examples of successful private insurance companies and the benefits they offer to customers. Remember to address the opposing viewpoint and refute their arguments.\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues in favor of a basic income, explaining why you believe it is a good idea and addressing the counterarguments presented in the claim. Your essay should be well-researched and supported by evidence, and should aim to persuade the reader that a basic income is a viable and beneficial policy option.\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues against the claim that Islam and Muslims are the new scapegoat for Western-led imperialism, mostly by America. Provide evidence and examples to support your argument. Consider discussing the complexities of international relations and the role of various factors in shaping Western foreign policies.\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues for the claim that in light of the increasing number of shootings in America, we should not publish the names or recognize individuals labeled as heroes in the same way we try to avoid doing with the shooter. Your essay should provide evidence and reasoning to support this claim.\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues the claim that a bank robbery that goes well is a victimless crime. Your essay should include a clear thesis statement, supporting evidence, and a conclusion that summarizes your argument.\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nsome universities offer online courses as an alternative to classes delivered on campus\nDo you think this is a positive or negative development\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues the claim that racism against African Americans will be greatly reduced only when income levels are similar to whites. Support your argument with evidence from at least three sources, and consider counterarguments to strengthen your position. Your essay should be at least 1000 words long.\n\nYour Writing:\n\nassistant\n",
        "system\nYou are a helpful assistant\nuser\nYou will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\nWrite an essay that argues against the claim that it's not a big deal for the FBI to bypass Apple's security measures. Your essay should provide evidence and reasoning to support your position. Be sure to address potential counterarguments and provide a strong conclusion that summarizes your main points.\n\nYour Writing:\n\nassistant\n",
    ]
    responses = [
        "The claim that the majority of insurance types should be managed by non-profit organizations or the government is a misleading and misguided notion. The idea that insurance should be completely controlled by the government or non-profits is a step towards a socialist or communist society where every aspect of society is run by a single entity. This approach eliminates competition and innovation in the insurance market, making it difficult for customers to get the best possible deal.\n\nOne of the main arguments against this claim is that private insurance companies are better equipped to provide insurance products and services to customers than non-profit organizations or government entities. Private insurance companies have the resources and expertise to develop and offer a wide range of insurance products that meet the diverse needs of their customers. Additionally, private insurance companies can offer competitive prices and offer better customer service than non-profit organizations or the government.\n\nAnother argument against this claim is that private insurance companies are better equipped to handle the complexities and risks associated with insurance. Insurance is a complex industry that involves a variety of risks and uncertainties. Private insurance companies have the experience and expertise to handle these risks and offer products that are tailored to meet the specific needs of their customers.\n\nFurthermore, the benefits that private insurance companies offer to customers cannot be ignored. Private insurance companies offer customers access to a wide range of insurance products, including health insurance, auto insurance, and life insurance. These companies also offer competitive prices and offer better customer service than non-profit organizations or the government.\n\nIn conclusion, the claim that the majority of insurance types should be managed by non-profit organizations or the government is a misguided and harmful idea. Private insurance companies are better equipped to provide insurance products and services to customers, offer better prices, and handle the complexities and risks associated with insurance. Therefore, it is important to support the private insurance industry and encourage competition and innovation in the insurance market.\n",
        "Introduction\n\nThe concept of a basic income has been discussed for many years, but it is only in recent years that it has begun to gain traction as a topic of debate. The idea of providing a regular and unconditional cash transfer to all citizens has been criticized as an expensive and ineffective policy, but it is also supported by many who believe that it could help to reduce poverty and improve the lives of those who currently struggle to make ends meet. In this essay, I will argue in favor of a basic income, explaining why I believe it is a good idea and addressing the counterarguments presented in the claim.\n\nArgument for a Basic Income\n\nThere are several reasons why a basic income is a good idea. One key benefit is that it can help to reduce poverty and inequality. Many individuals who struggle to make ends meet are currently reliant on welfare or other forms of government assistance, which can be expensive and often insufficient to cover all of their needs. A basic income can provide a guaranteed income to all citizens, ensuring that they have enough money to meet their basic needs, such as food, shelter, and healthcare. This can help to reduce poverty and inequality, as well as improve overall well-being and quality of life for many individuals.\n\nAnother benefit of a basic income is that it can help to promote work ethic and reduce unemployment. Many individuals who are struggling to make ends meet are also struggling to find work, which can be a vicious cycle of poverty and unemployment. A basic income can provide a stable income that encourages individuals to seek work and build skills and knowledge, which can lead to better employment opportunities and higher incomes in the long run.\n\nHowever, there are also counterarguments to the idea of a basic income. One common argument is that it is too expensive to implement. While there may be upfront costs associated with setting up a basic income program, it is often argued that the potential benefits would outweigh the costs over the long run. For example, a study by the Intergenerational Poverty Reduction Centre found",
        "Title: Islam and Muslims: The New Scapegoat for Western-led Imperialism?\n\nIn the recent past, Islam and Muslims have become the scapegoat for Western-led imperialism. Many Western nations, especially the United States, have accused Muslims and Islam of being responsible for the world's problems, including terrorism and poverty. However, this is a flawed argument that ignores the complexities of international relations and the role of various factors in shaping Western foreign policies.\n\nFirstly, it is important to note that terrorism is not limited to any one religion. In fact, terrorism is used as a tool by extremist groups to promote their agenda. Therefore, attributing terrorism to Islam and Muslims is a false dichotomy that ignores the fact that terrorism is a global phenomenon.\n\nSecondly, the idea that Islam and Muslims are responsible for poverty is also a fallacy. Poverty is a complex issue that is rooted in various factors such as economic structures, political instability, and lack of access to education and healthcare. It is not possible to pin the blame for poverty solely on any one religion or group.\n\nFinally, the accusation that Islam and Muslims are responsible for imperialism is also not justified. Western-led imperialism has been a global phenomenon for centuries, and it has shaped the course of history in many ways. The West has played a significant role in shaping the world, and it is not fair to single out Islam and Muslims as the cause of imperialism.\n\nFurthermore, the idea that Islam and Muslims are the new scapegoat for Western-led imperialism is not backed by evidence. There is no concrete evidence that shows that the actions of Western-led imperialism have been influenced by Islam or Muslims. In fact, some Western countries have even invested significant resources in promoting Islam and Muslims as part of their foreign policies.\n\nIn conclusion, it is important to recognize that Islam and Muslims are not the new scapegoat for Western-led imperialism. The accusations against them are based on false premises and do not take into account the complexities of international relations and the",
        "The increasing number of shootings in America has led to a debate on how to treat individuals who are labeled as heroes. While acknowledging the bravery of these individuals may be important, it is equally important to avoid glorifying them. This is because acknowledging their actions may be seen as giving them undue attention and could lead to dangerous behavior. \n\nInstead of promoting these individuals as heroes, we should focus on promoting public safety. This can be achieved by implementing stricter gun control laws, increasing police patrols, and providing resources to help those in need. By taking these steps, we can create a more secure and safe society, where individuals are not afraid to speak out against injustice and violence. \n\nFurthermore, acknowledging individuals as heroes can lead to a false sense of security. This can cause people to become complacent, and not take appropriate measures to prevent further tragedies. By promoting public safety, we can create a culture of vigilance and preparedness, where individuals are more likely to take action to prevent violence. \n\nIn conclusion, while acknowledging the bravery of individuals who are labeled as heroes may be important, it is equally important not to glorify them. Instead, we should focus on promoting public safety by implementing stricter gun control laws, increasing police patrols, and providing resources to help those in need. By doing so, we can create a more secure and safe society, where individuals are not afraid to speak out against injustice and violence.\n",
        "The genre of the essay is an argumentative essay, which is a type of essay that presents a writer's viewpoint on a topic. The topic of this essay is the claim that a bank robbery that goes well is a victimless crime.\n\nThesis statement: The claim that a bank robbery that goes well is a victimless crime is false.\n\nSupporting evidence:\n\n1. The financial consequences of a bank robbery can be severe and devastating to the victims, including loss of money, assets, and personal property.\n\n2. The impact of a bank robbery on the victims can be long-lasting, as it can lead to mental and emotional trauma, financial instability, and even the loss of life.\n\n3. The financial consequences of a bank robbery can also affect the victims' ability to access financial resources, leading to further financial instability and poverty.\n\n4. The victims of a bank robbery may also face legal consequences, which can lead to financial and emotional distress, as well as the loss of their freedom.\n\nConclusion:\n\nThe claim that a bank robbery that goes well is a victimless crime is false. The financial consequences of a bank robbery can be severe and devastating to the victims, including loss of money, assets, and personal property. The impact of a bank robbery on the victims can be long-lasting, leading to mental and emotional trauma, financial instability, and even the loss of life. The financial consequences of a bank robbery can also affect the victims' ability to access financial resources, leading to further financial instability and poverty. Additionally, the victims of a bank robbery may also face legal consequences, which can lead to financial and emotional distress, as well as the loss of their freedom. Therefore, it is essential to take steps to prevent bank robberies and to hold those responsible for such crimes accountable.\n",
        "The rise of online courses has brought both advantages and disadvantages to universities and their students. On one hand, online courses offer flexibility, allowing students to complete their studies from any location with an internet connection. This can be particularly beneficial for students who are unable to attend classes on campus due to work, family, or personal reasons. Online courses also provide students with access to a wider range of courses and instructors, as well as more affordable options compared to traditional campus-based courses.\n\nOn the other hand, online courses have their drawbacks. Some students may struggle to stay motivated and engaged in online learning, as the absence of the physical presence of other students and instructors can make it difficult to stay focused. Online courses may also lack the personal feedback and interaction that students may receive in traditional classroom settings. Moreover, the lack of social and networking opportunities may limit the potential for networking and professional development.\n\nOverall, while online courses offer many benefits, they should not be seen as a complete replacement for traditional campus-based courses. Rather, they should be viewed as a complementary tool that can be used in conjunction with traditional courses to provide students with the best learning experience possible.\n",
        "The essay is not provided in the given instructions.\n",
        "The FBI's decision to bypass Apple's security measures has been widely criticized as being unnecessary and possibly illegal. However, I believe that it is not a big deal for the FBI to do so. This is because the technology used by Apple's devices is not as secure as we think. In fact, the NSA has been able to successfully exploit vulnerabilities in Apple's devices in the past. Additionally, the FBI is not the only agency that has access to the technology used by Apple's devices. The National Security Agency (NSA) has also been able to access the same technology. Therefore, it is not a big deal for the FBI to bypass Apple's security measures.\n\nHowever, some argue that the FBI's actions are not legal and could lead to a breach of privacy. In response, I believe that the FBI is within its rights to bypass Apple's security measures. The Constitution of the United States guarantees the right to privacy, but it does not guarantee the right to absolute privacy. In fact, the government has a duty to protect the public from harm, which could be achieved through the use of technology that is necessary to do so.\n\nFurthermore, I believe that it is not a big deal for the FBI to bypass Apple's security measures because it is not a significant breach of privacy. The technology used by Apple's devices is not as secure as we think, and the NSA has been able to successfully exploit vulnerabilities in Apple's devices in the past. Therefore, it is not a big deal for the FBI to bypass Apple's security measures, as it is not a significant breach of privacy.\n\nIn conclusion, I believe that the FBI's decision to bypass Apple's security measures is not a big deal. The technology used by Apple's devices is not as secure as we think, and the NSA has been able to successfully exploit vulnerabilities in Apple's devices in the past. Additionally, the FBI is not the only agency that has access to the technology used by Apple's devices. Therefore, it is not",
    ]

    scores, evaluations = reward_model.score(queries, responses)
    for s, e in zip(scores, evaluations):
        print(f"Score: {s}")
        print(f"Evaluations: {e}")
        print("-" * 50)

    # meta_reward_model = MetaRewardModelEssayWriting(
    #     reward_model_address=reward_model_address,
    #     experiment_directory=experiment_directory,
    # )
    # evaluation_results = meta_reward_model.meta_evaluate_and_update(batch_index=batch_index, return_evaluations=True)
    # with open(os.path.join(prompts_directory, f"meta_evaluation_results_{batch_index}.json"), "w") as f:
    #     json.dump(evaluation_results, f, indent=4)
