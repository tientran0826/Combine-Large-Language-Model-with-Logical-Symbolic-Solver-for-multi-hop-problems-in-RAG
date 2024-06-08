import pandas as pd
import numpy as np
import re
from nltk.inference import Prover9Command
from nltk.inference.prover9 import *
import json
import toml


def read_api_key(file_path, openai=False):
    try:
        with open(file_path, "r") as file:
            config = toml.load(file)
            api_keys = config.get("api_keys", {})
            if openai: return api_keys.get("OPENAI_KEY")
            return api_keys.get("API_KEY")
    except FileNotFoundError:
        print("Config file not found.")
        return None


def extract_logic_statements(text):
    lines = text.split('\n')
    premises = []
    conclusion = ""
    in_premises = False
    in_conclusion = False

    for line in lines:
        line = line.strip()
        if line.startswith("Premises:"):
            in_premises = True
            continue
        elif line.startswith("Conclusion:"):
            in_premises = False
            in_conclusion = True
            continue

        if in_premises:
            if line:
                premises.append(line)
        elif in_conclusion:
            if line:
                conclusion = line

    return premises, conclusion


def extract_premises_and_conclusion(text):
    premises_pattern = re.compile(r"Premises:\s*\[([^\]]+)\]", re.DOTALL)
    conclusion_pattern = re.compile(r"Conclusion:\s*(.*)")

    premises_match = premises_pattern.search(text)
    premises = []
    if premises_match:
        premises_text = premises_match.group(1).strip()
        premises = [p.strip().strip("'") for p in premises_text.split(',')]

    conclusion_match = conclusion_pattern.search(text)
    conclusion = ""
    if conclusion_match:
        conclusion = conclusion_match.group(1).strip()

    return premises, conclusion


def prove_arguments(dataset):
    error_count = 0
    result_df = []
    for index, row in dataset.iterrows():

        result = None
        has_error = None
        try:
            conclusion_without_period = row[f'conclusion'].replace(".", "")
            premises_without_periods = [
                premise.replace(".", "") for premise in row[f'premises']
            ]
            argument = (conclusion_without_period, premises_without_periods)
            goal, assumptions = argument
            g = Expression.fromstring(goal)
            alist = [Expression.fromstring(a) for a in assumptions]
            p = Prover9Command(g, assumptions=alist).prove()
            print(f"Argument {index}:")
            for a in alist:
                print("   %s" % a)
            print(f"==> {g}: {p}\n")
            result = p

        except Exception as e:
            print(f"Error in argument {index}: {e}")
            error_count = error_count + 1
            has_error = e
        row["prover9_result"] = result
        row["has_error"] = has_error
        result_df.append(row)
    print(f"Total errors: {error_count}/{len(dataset)}")
    return pd.DataFrame(result_df)


def prove(argument):
    try:
        goal, assumptions = argument
        g = Expression.fromstring(goal)
        alist = [Expression.fromstring(a) for a in assumptions]
        p = Prover9Command(g, assumptions=alist).prove()
        return None, p
    except Exception as e:
        return e, None
