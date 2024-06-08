import argparse
import json
import os
import openai
from tqdm import tqdm
from utils import extract_premises_and_conclusion, prove, read_api_key

# Read the API key from the configuration file
api_key = read_api_key('pyproject.toml', openai=True)

os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'


class ErrorRefiner:

    def __init__(self,
                 prompt_path,
                 json_path,
                 api,
                 model_name,
                 request_limit=2,
                 output_folder=None):
        self.prompt_path = prompt_path
        self.json_path = json_path
        self.model_name = model_name
        self.request_limit = request_limit
        self.output_folder = output_folder
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=api_key) if 'gpt' not in model_name else openai.OpenAI(
                api_key=api_key)

    def refine_errors(self):
        # Read the prompt template
        with open(self.prompt_path, 'r') as f:
            refine_prompt = f.read()

        # Read the JSON data
        with open(self.json_path, 'r') as file:
            json_data = json.load(file)

        refined_data = []

        # Iterate through each data entry
        for index, data in tqdm(enumerate(json_data), total=len(json_data)):
            if data['has_error']:
                premises, conclusion, error = data['premises'], data[
                    'conclusion'], data['has_error']
                correct_try = 0

                while correct_try < self.request_limit:
                    # Prepare the prompt for the model
                    prompt = refine_prompt.replace("[[PREMISES]]",
                                                   str(premises))
                    prompt = prompt.replace("[[CONCLUSION]]", conclusion)
                    prompt = prompt.replace("[[ERROR MESSAGES]]", str(error))

                    messages = [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": prompt
                    }]

                    # Get response from the model
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=1,
                    )

                    # Extract the corrected formula from the response
                    correct_formula = response.choices[0].message.content
                    premises, conclusion = extract_premises_and_conclusion(
                        correct_formula)

                    # Clean periods for the proof argument
                    conclusion_without_period = conclusion.replace(".", "")
                    premises_without_periods = [
                        premise.replace(".", "") for premise in premises
                    ]
                    argument = (conclusion_without_period,
                                premises_without_periods)

                    # Prove the corrected formula
                    error, is_proof = prove(argument)

                    if is_proof:
                        # Update the data with corrected premises and conclusion
                        data['premises'] = premises
                        data['conclusion'] = conclusion
                        data["prover9_result"] = str(is_proof)
                        data["has_error"] = None
                        break
                    else:
                        data["has_error"] = error

                    correct_try += 1

            refined_data.append(data)

        return refined_data


def main():
    # Setup argument parser
    model_name = 'gpt-3.5-turbo'
    model_name_path = model_name
    if 'Llama-3-8B' in model_name: model_name_path = 'Llama-3-8B'
    elif 'Llama-3-70B' in model_name: model_name_path = 'Llama-3-70B'
    elif 'Mixtral-8x22B' in model_name:
        model_name_path = 'Mixtral-8x22B'

    parser = argparse.ArgumentParser(description='Error Refiner')
    parser.add_argument('--prompt_path',
                        type=str,
                        default='./models/prompts/self-correct.txt',
                        help='Path to the prompt file')
    parser.add_argument(
        '--json_path',
        type=str,
        default=f'./outputs/logic_inference/results_{model_name_path}.json',
        help='Path to the JSON file')
    parser.add_argument('--api',
                        type=str,
                        default=api_key,
                        help='API key to use')
    parser.add_argument('--model_name',
                        default=model_name,
                        type=str,
                        help='Name of the model')
    parser.add_argument('--output_folder',
                        type=str,
                        default='./outputs/self-refinement/',
                        help='Path to the output folder')
    args = parser.parse_args()

    # Initialize the ErrorRefiner
    refiner = ErrorRefiner(args.prompt_path,
                           args.json_path,
                           args.api,
                           args.model_name,
                           output_folder=args.output_folder)
    refined_result_df = refiner.refine_errors()
    # Write the refined data to the output folder
    if args.output_folder:

        output_path = os.path.join(args.output_folder,
                                   f'refined_data_{model_name_path}.json')
        with open(output_path, 'w') as f:
            json.dump(refined_result_df, f, default=str,
                      indent=4)  # Ensure all objects are JSON serializable


if __name__ == "__main__":
    main()
