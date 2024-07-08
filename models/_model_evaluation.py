import argparse
import pandas as pd
import numpy as np
import openai
import json
import os
from tqdm import tqdm
from utils import read_api_key

api_key = read_api_key('pyproject.toml')
model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
if 'Llama-3-8B' in model_name: model_name_path = 'Llama-3-8B'
elif 'Llama-3-70B' in model_name: model_name_path = 'Llama-3-70B'
elif 'Mixtral-8x22B' in model_name:
    model_name_path = 'Mixtral-8x22B'

class ModelEvaluation:

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
            
    def get_answer_model(self, model_answer, answer):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace("[[MODEL ANSWER]]", model_answer)
        prompt = prompt.replace("[[ANSWER]]", answer)
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

    def extract_scores(self, text):
        import re
        parts = text.split('\n')
        result_dict = {}

        for part in parts:
            key, value = part.split(': ')
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
            result_dict[key] = value
        return result_dict
    
    def process_row(self, row):
        answer = self.get_answer_model(row['model_answer'],
                                      row['answer_text'],)
        answer = self.extract_scores(answer)
        print(answer)
        return answer

    def process_dataset(self):
        result_data = pd.read_json(self.input_datasets[0])

        data = pd.read_json(self.input_datasets[1], lines=True)
        data = data[['_id', 'answer_text']].rename(columns={'_id':'id'})
        # Merge datasets on 'Id'
        merged_df = pd.merge(result_data, data, on='id')
    
        results = []
        for _, row in tqdm(merged_df.iterrows(),
                           total=len(merged_df),
                           desc="Processing rows"):
            try:
                processed_row = self.process_row(row)
                results.append(processed_row)
            except Exception as e:
                print(e)
                
        result_df = pd.DataFrame(results)
        result_json = {'Correct Score': np.mean(result_df['Correct Score']),
                       'Inference Score': np.mean(result_df['Inference Score'])}
        print(result_json)
        if 'Llama-3-8B' in self.model_name: self.model_name = 'Llama-3-8B'
        elif 'Llama-3-70B' in self.model_name: self.model_name = 'Llama-3-70B'
        elif 'Mixtral-8x22B' in self.model_name:
            self.model_name = 'Mixtral-8x22B'
        output_path = os.path.join(self.output_path,
                                   f"model_evaluation_{self.model_name}.json")
        with open(output_path, 'w') as json_file:
                json.dump(result_json, json_file, indent=4)

def main():
    print("Model Evaluation: \n")
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
                        default='./models/prompts/model-evaluation.txt',
                        help='Path to the prompt file (default: %(default)s)')
    parser.add_argument(
        '--input_datasets',
        nargs=2,
        type=str,
        default=[f'./outputs/answer_result/answer_result_{model_name_path}.json', f'./data/en_legal_question_answer_dataset.jsonl'],
        help='Paths to the input datasets (default: %(default)s)')
    parser.add_argument('--output_path',
                        type=str,
                        default='./outputs/answer_evaluation')
    args = parser.parse_args()
    processor = ModelEvaluation(args.api_key, args.model_name, args.prompt_path,
                             args.input_datasets, args.output_path)
    processor.process_dataset()


if __name__ == '__main__':
    main()
