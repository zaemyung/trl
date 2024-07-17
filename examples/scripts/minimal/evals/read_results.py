import json
import pandas as pd

with open("ppo_rlaif.json", "r") as f:
    df = pd.DataFrame.from_dict(json.load(f))

for index, row in df.iterrows():
    print(f'PROMPT:\n{row["prompt"]}\n##################\n')
    print(f'rlaif model:\n{row["model_response"]}\n##################\n')
    print(f'ref model:\n{row["reference_response"]}\n##################\n')
    input("next")
