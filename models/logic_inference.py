import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from nltk.sem import Expression
from nltk.inference import Prover9Command

# Ensure Prover9's executable path is set
os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'
model_name = "gpt-3.5-turbo"

# Adjust model name if it matches certain patterns
model_name = {
    'Llama-3-8B': 'Llama-3-8B',
    'Llama-3-70B': 'Llama-3-70B',
    'Mixtral-8x22B': 'Mixtral-8x22B'
}.get(model_name, model_name)


class LogicInference:
    def __init__(self):
        pass

    def prove_arguments_from_json(self, json_path, output_folder):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        error_count = 0
        result_list = []

        for index, data in tqdm(enumerate(json_data), total=len(json_data), desc="Proving arguments"):
            print("=" * 66)
            result = None
            has_error = None
            try:
                conclusion = data['conclusion'].replace(".", "")
                premises = [premise.replace(".", "") for premise in data['premises']]
                goal = Expression.fromstring(conclusion)
                assumptions = [Expression.fromstring(premise) for premise in premises]
                prover_result = Prover9Command(goal, assumptions=assumptions).prove()

                print(f"Argument {index}:")
                for assumption in assumptions:
                    print(f"   {assumption}")
                print(f"==> {goal}: {prover_result}\n")

                result = prover_result

            except Exception as e:
                print(f"Error in argument {index}: {e}")
                error_count += 1
                has_error = str(e)

            result_dict = {
                "id": data.get("id", ""),
                "conclusion": data.get("conclusion", ""),
                "premises": data.get("premises", []),
                "prover9_result": str(result),
                "has_error": has_error
            }
            result_list.append(result_dict)

        print(f"Total errors: {error_count}/{len(json_data)}")

        output_path = os.path.join(output_folder, f"results_{model_name}.json")
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, 'w') as outfile:
            json.dump(result_list, outfile, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Perform logic inference on arguments provided in JSON format.')
    parser.add_argument('--json_path', type=str, default=f'./outputs/logic_programs/logic_program_{model_name}.json',
                        help='Path to the JSON file containing logic programs. Default is ./outputs/logic_programs/logic_program.json.')
    parser.add_argument('--output_path', type=str, default='./outputs/logic_inference',
                        help='Path to the directory where the output will be saved. Default is ./outputs/logic_inference.')

    args = parser.parse_args()

    logic_inference = LogicInference()
    logic_inference.prove_arguments_from_json(args.json_path, args.output_path)


if __name__ == '__main__':
    main()
