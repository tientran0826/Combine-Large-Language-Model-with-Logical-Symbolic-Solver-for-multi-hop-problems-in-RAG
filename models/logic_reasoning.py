import argparse
import pandas as pd
import openai
import json
import os
from tqdm import tqdm
from utils import read_api_key

api_key = read_api_key('pyproject.toml')
model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
model_name_path = 'gpt-3.5-turbo'

class LogicReasoning:

    def __init__(self, api_key, model_name, prompt_path, input_datasets,
                 output_path):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.input_datasets = input_datasets
        self.output_path = output_path
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=api_key) if 'gpt' not in model_name else openai.OpenAI(
                api_key=api_key)
            
    def get_answer_model(self, raw, premises, conclusion, prover9_result):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace("[[RAW]]", raw)
        prompt = prompt.replace("[[PREMISES]]", str(premises))
        prompt = prompt.replace("[[CONCLUSION]]", conclusion)
        prompt = prompt.replace("[[PROVER9_RESULT]]", prover9_result)
        messages = [{
            "role": "system",
            "content": "You are a law assistant."
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
        answer = self.get_answer_model(row['raw'],
                                      row['premises'],
                                      row['conclusion'],
                                      row['prover9_result'])
        answer = answer.replace("RESULT EXPLAIN:", "").replace("Here is the explanation:", "")
        print(answer)
        return {'model_answer': answer}

    def process_dataset(self):
        refine_data = pd.read_json(self.input_datasets[0])
        refine_data = refine_data[['id','premises','conclusion','prover9_result','has_error']]
        raw_data = pd.read_json(self.input_datasets[1])
        raw_data = raw_data[['id', 'raw']]
        # Merge datasets on 'Id'
        merged_df = pd.merge(refine_data, raw_data, on='id')
    
        json_list = []
        for _, row in tqdm(merged_df.iterrows(),
                           total=len(merged_df),
                           desc="Processing rows"):
            try:
                processed_row = self.process_row(row)
                processed_row['id'] = row['id']
                json_list.append(processed_row)
            except Exception as e:
                print(e)
        output_path = os.path.join(self.output_path,
                                   f"answer_result_{model_name_path}.json")
        with open(output_path, 'w') as json_file:
            json.dump(json_list, json_file, indent=4)


def main():
    print("Deductive reasoning: \n")
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
                        default='./models/prompts/explain-result.txt',
                        help='Path to the prompt file (default: %(default)s)')
    parser.add_argument(
        '--input_datasets',
        nargs=2,
        type=str,
        default=[f'./outputs/self-refinement/refined_data_{model_name_path}.json', f'./outputs/logic_programs/logic_program_{model_name_path}.json'],
        help='Paths to the input datasets (default: %(default)s)')
    parser.add_argument('--output_path',
                        type=str,
                        default='./outputs/answer_result')
    args = parser.parse_args()
    processor = LogicReasoning(args.api_key, args.model_name, args.prompt_path,
                             args.input_datasets, args.output_path)
    processor.process_dataset()


if __name__ == '__main__':
    main()
