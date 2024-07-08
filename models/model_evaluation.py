import argparse
import pandas as pd
import numpy as np
import openai
import json
import os
from tqdm import tqdm
from utils import read_api_key
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
#from deepeval.models import ChatOpenAI  # Ensure you import ChatOpenAI
import os
from deepeval.models.base_model import DeepEvalBaseLLM

api_key = read_api_key('pyproject.toml')
# os.environ["OPENAI_API_KEY"] = api_key
model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
model_name_path = 'gpt-3.5-turbo'
# if 'Llama-3-8B' in model_name: model_name_path = 'Llama-3-8B'
# elif 'Llama-3-70B' in model_name: model_name_path = 'Llama-3-70B'
# elif 'Mixtral-8x22B' in model_name:
#     model_name_path = 'Mixtral-8x22B'
print(model_name_path)
class AnyScaleAI(DeepEvalBaseLLM):
    def __init__(self):
        self.model = model_name
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=api_key
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        messages = [{
            "role": "system",
            "content": "You are a law assistant."
        }, {
            "role": "user",
            "content": prompt
        }]

        response = self.client.chat.completions.create(
            model=chat_model,
            messages=messages
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt) 

    def get_model_name(self):
        return "AnyScale Model"
    

class ModelEvaluation:

    def __init__(self, input_datasets, output_path):

        self.input_datasets = input_datasets
        self.output_path = output_path
        self.model = AnyScaleAI()
        self.correctness_metric = GEval(
            model=self.model,
            name="Legal Correctness",
            evaluation_steps = [
                "If the model does not have a conclusion for this question, it means that it can be uncertain. Check if it provides the correct conditions to make the question certain.",
                "Identify if the model correctly states the principles relevant to the field being considered.",
                "Verify if the model accurately describes the conditions under which individuals or entities can perform actions.",
                "Check if the model includes all relevant points, such as categories of individuals or special conditions.",
                "Ensure that the model mentions key details and conditions important to the situation being considered.",
                "Compare the model's conclusion with the expert's answer to see if they align in terms of reasoning and outcome.",
                "Verify if the model avoids contradicting the provisions or principles cited by the expert."
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
        )

        self.inference_metric = GEval(
            model=self.model,
            name="Legal Inference",
            evaluation_steps = [
                "Assess if the model's explanation of relevant principles is clear and easy to understand.",
                "Determine if the explanation is straightforward and uses accessible language.",
                "Evaluate if the model provides an adequate explanation of the relevant context, including who qualifies and under what circumstances.",
                "Check if the model provides enough context for the reader to understand the implications of the regulations or principles.",
                "Verify if the model correctly uses terms and jargon appropriate to the context.",
                "Ensure that the terms are used accurately to convey the correct concepts."
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
        )

    def create_llm_test_case(self, row):
        # Customize this function based on your dataset structure
        input_text = row['question_text']
        model_answer = row['model_answer']
        expected_answer = row['answer_text']

        # Create LLMTestCase instance
        test_case = LLMTestCase(
            input=input_text,
            actual_output=model_answer,  # Use model answer directly
            expected_output=expected_answer)

        return test_case

    def process_dataset(self, include_error=True):
        result_data = pd.read_json(self.input_datasets[0])
        # error_data = pd.read_json(f'./outputs/self-refinement/refined_data_{model_name_path}.json')
        # error_data = error_data[error_data['has_error'].isna()].rename(columns={'id':'Id'})['Id']    
        data = pd.read_json(self.input_datasets[1], lines=True)
        data = data[['_id', 'question_text', 'answer_text']].rename(columns={'_id':'id'})
        # Merge datasets on 'Id'
        merged_df = pd.merge(result_data, data, on='id')
        #merged_df = pd.merge(error_data, merged_df, on='Id')
        #print(merged_df)
        
        results = []
        for _, row in tqdm(merged_df.iterrows(),
                           total=len(merged_df),
                           desc="Processing rows"):
            try:
                test_case = self.create_llm_test_case(row)
                # Measure test case using metrics
                self.correctness_metric.measure(test_case)
                self.inference_metric.measure(test_case)

                # Collect results if needed
                result = {
                    'Id': row['id'],
                    'Correct Score': self.correctness_metric.score,
                    'Correct Reason': self.correctness_metric.reason,
                    'Inference Score': self.inference_metric.score,
                    'Inference Reason': self.inference_metric.reason
                    
                }
                results.append(result)
                print("="*10)
                print(result)
            except Exception as e:
                print(f"Error processing row {row['id']}: {str(e)}")

        # Calculate averages
        if results:
            average_correct_score = np.mean(
                [res['Correct Score'] for res in results])
            average_inference_score = np.mean(
                [res['Inference Score'] for res in results])
            result_json = {
                'Average Correct Score': average_correct_score,
                'Average Inference Score': average_inference_score
            }

            # Save results to JSON file
            output_file = f"model_evaluation_{model_name_path}.json"
            output_full = f"model_evaluation_{model_name_path}_full.json"
            output_path = os.path.join(self.output_path, output_file)
            output_path_full = os.path.join(self.output_path, output_full)
            with open(output_path, 'w') as json_file:
                json.dump(result_json, json_file, indent=4)
            with open(output_path_full, 'w') as json_file:
                json.dump(results, json_file, indent=4)
        else:
            print("No results to process.")


def main():
    print("Model Evaluation: \n")
    parser = argparse.ArgumentParser(
        description='Process some arguments for Prover9Processor.')
    parser.add_argument(
        '--input_datasets',
        nargs=2,
        type=str,
        default=[
            f'./outputs/answer_result/SymbCoT/answer_result_{model_name_path}.json',
            './data/en_legal_question_answer_dataset.jsonl'
        ],
        help='Paths to the input datasets (default: %(default)s)')
    parser.add_argument('--output_path',
                        type=str,
                        default=f'./outputs/answer_evaluation/')
    args = parser.parse_args()

    processor = ModelEvaluation(args.input_datasets, args.output_path)
    processor.process_dataset()


if __name__ == '__main__':
    main()
