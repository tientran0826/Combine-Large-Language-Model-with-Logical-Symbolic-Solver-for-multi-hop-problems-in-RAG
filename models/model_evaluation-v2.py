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
linc_model = 'Llama-3-70B'
cot_model = 'gpt-3.5-turbo'

class ModelEvaluation:

    def __init__(self, api_key, model_name, prompt_path,  model_answers, input_dataset, output_path):
        self.api_key = api_key
        self.model_name = model_name
        self.model_answers = model_answers
        self.prompt_path = prompt_path
        self.input_dataset = input_dataset
        self.output_path = output_path
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=api_key) if 'gpt' not in model_name else openai.OpenAI(
                api_key=api_key)
            
    def get_answer_model(self, question, model_answer_1, model_answer_2):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace("[[QUESTION]]", question)
        prompt = prompt.replace("[[ANSWER 1]]", model_answer_1)
        prompt = prompt.replace("[[ANSWER 2]]", model_answer_2)
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

        lines = text.split('\n')

        # Initialize variables to store scores
        score_answer1 = None
        score_answer2 = None

        # Iterate through each line to find scores
        for line in lines:
                if line.startswith('+ Answer 1:'):
                    score_answer1 =  float(line.split(': ')[1])
                elif line.startswith('+ Answer 2:'):
                    score_answer2 =  float(line.split(': ')[1])

        return score_answer1, score_answer2
    def process_row(self, row):
        answer = self.get_answer_model( row['Question'],
                                        row['Answer 1'],
                                        row['Answer 2'],)
        score_answer1, score_answer2 = self.extract_scores(answer)
        return score_answer1, score_answer2

    def process_dataset(self):
        dataset = pd.read_json(self.input_dataset, lines=True)[['Id', 'Question']].rename(columns={'Id':'id'})
        cot_data = pd.read_json(self.model_answers[0], lines=True).rename(columns={'answer': 'Answer 1', 'Id':'id'})
        linc_data = pd.read_json(self.model_answers[1]).rename(columns={'model_answer': 'Answer 2'})
        # Merge datasets on 'Id'
        merged_df = pd.merge(cot_data, linc_data, on='id')
        merged_df = pd.merge(dataset, merged_df, on='id')
        results = []
        for _, row in tqdm(merged_df.iterrows(),
                           total=len(merged_df),
                           desc="Processing rows"):
            try:
                processed_row = self.process_row(row)
                print(processed_row)
                results.append(processed_row)
            except Exception as e:
                print(e)
                
        result_df = pd.DataFrame(results, columns=['Answer 1', 'Answer 2'])
        print(result_df)
        result_json = {f'CoT {cot_model} Score': np.mean(result_df['Answer 1']),
                       f'Linc {linc_model} Score ': np.mean(result_df['Answer 2']),}
        print(result_json)
        output_path = os.path.join(self.output_path,
                                   f"model_evaluation_{cot_model}_{linc_model}.json")
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
                        default='./models/prompts/model-evaluation-v2.txt',
                        help='Path to the prompt file (default: %(default)s)')
    
    parser.add_argument(
        '--input_dataset',
        nargs=2,
        type=str,
        default=f'./data/fol_dataset.jsonl',
        help='Paths to the input dataset (default: %(default)s)')
    

    parser.add_argument(
        '--input_model_answer',
        nargs=2,
        type=str,
        default=[f'./baselines/outputs/CoT/result_{cot_model}.jsonl', 
                 f'./outputs/answer_result/answer_result_{linc_model}.json'],
        help='Paths to the model answers (default: %(default)s)')
    parser.add_argument('--output_path',
                        type=str,
                        default='./outputs/answer_evaluation')
    args = parser.parse_args()
    processor = ModelEvaluation(args.api_key, args.model_name, args.prompt_path, args.input_model_answer,
                             args.input_dataset, args.output_path)
    processor.process_dataset()


if __name__ == '__main__':
    main()
