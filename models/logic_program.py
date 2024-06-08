import argparse
import pandas as pd
import openai
import json, os
from tqdm import tqdm
from utils import read_api_key

api_key = read_api_key('pyproject.toml', openai=True)
model_name = 'gpt-3.5-turbo'


class LogicProgram:

    def __init__(self, api_key, model_name, prompt_path, input_dataset,
                 output_path):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.input_dataset = input_dataset
        self.output_path = output_path
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=api_key) if 'gpt' not in model_name else openai.OpenAI(
                api_key=api_key)

    def extract_logic_statements(self, text):
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

    def get_prover9_formula_from_llm(self, rules, goal):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()

        prompt = prompt.replace("[Rules]", rules)
        prompt = prompt.replace("[Goals]", goal)

        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": prompt
        }]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content

    def process_row(self, row):
        result = self.get_prover9_formula_from_llm(row['Law Context'],
                                                   row['Question'])
        premises, conclusion = self.extract_logic_statements(result)
        return {'raw': result, 'premises': premises, 'conclusion': conclusion}

    def process_dataset(self):
        df = pd.read_json(self.input_dataset)
        json_list = []
        for _, row in tqdm(df.iterrows(),
                           total=len(df),
                           desc="Processing rows"):
            try:
                processed_row = self.process_row(row)
                processed_row['id'] = row['Id']
                json_list.append(processed_row)
            except Exception as e:
                print(e)
        if 'Llama-3-8B' in self.model_name: self.model_name = 'Llama-3-8B'
        elif 'Llama-3-70B' in self.model_name: self.model_name = 'Llama-3-70B'
        elif 'Mixtral-8x22B' in self.model_name:
            self.model_name = 'Mixtral-8x22B'
        output_path = os.path.join(self.output_path,
                                   f"logic_program_{self.model_name}.json")
        with open(output_path, 'w') as json_file:
            json.dump(json_list, json_file, indent=4)


def main():
    print("Logic Program: \n")
    parser = argparse.ArgumentParser(
        description='Process some arguments for Prover9Processor.')
    parser.add_argument(
        '--api_key',
        type=str,
        default=api_key,
        help='API key for accessing the model (default: %(default)s)')
    parser.add_argument(
        '--model_name',
        type=str,
        default=model_name,
        help='Name of the model to be used (default: %(default)s)')
    parser.add_argument('--prompt_path',
                        type=str,
                        default='./models/prompts/prover9-parsrer.txt',
                        help='Path to the prompt file (default: %(default)s)')
    parser.add_argument(
        '--input_dataset',
        type=str,
        default='./data/dataset.json',
        help='Path to the input dataset (default: %(default)s)')
    parser.add_argument('--output_path',
                        type=str,
                        default='./outputs/logic_programs')
    args = parser.parse_args()
    processor = LogicProgram(args.api_key, args.model_name, args.prompt_path,
                             args.input_dataset, args.output_path)
    processor.process_dataset()


if __name__ == '__main__':
    main()
