import pandas as pd
import json
import openai
import toml
import os
from tqdm import tqdm  # Import tqdm for progress tracking

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

def get_model_name_path(model_name):
    if 'Llama-3-8B' in model_name:
        return 'Llama-3-8B'
    elif 'Llama-3-70B' in model_name:
        return 'Llama-3-70B'
    elif 'Mixtral-8x22B' in model_name:
        return 'Mixtral-8x22B'
    elif 'CodeLlama-34b' in model_name:
        return 'CodeLlama-34b'
    return model_name

api_key = read_api_key('pyproject.toml', openai=True)
print(api_key)
# TURN IT ON WHEN USING OPENAI API
# client = openai.OpenAI(base_url="https://api.endpoints.anyscale.com/v1",
#                        api_key=api_key)
client = openai.OpenAI(api_key=api_key)

cot_prompt_path = './baselines/prompts/cot.txt'
with open(cot_prompt_path, 'r') as file:
    cot_prompt = file.read()

def get_cot_from_llm(model_name, prompt, rules, question):
    prompt = prompt.replace("[Rules]", rules)
    prompt = prompt.replace("[Question]", question)
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": prompt
    }]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content

def process_row(row, model_name):
    raw_result = get_cot_from_llm(model_name, cot_prompt,
                                    row['Law Context'], row['Question'])
    return pd.Series({'answer': raw_result})

# Read the JSON file into a DataFrame
df = pd.read_json("./data/fol_dataset.jsonl", lines=True)

# List of models to apply
model_names = [
   "gpt-3.5-turbo"
]

# Iterate through each model name and process rows
for model_name in model_names:
    model_name_path = get_model_name_path(model_name)
    print(f"Processing model: {model_name}")
    
    # Use tqdm to track progress
    tqdm.pandas()
    results = df.progress_apply(process_row, axis=1, args=(model_name, ))
    
    df['answer'] = results['answer']
    result_df = df[['Id', 'answer']]
    output_path = os.path.join('./baselines/outputs/CoT',
                               f'result_{model_name_path}.jsonl')
    result_df.to_json(output_path, orient='records', lines=True)
    
    print(f"Processing completed for model: {model_name}")

print("All models processed.")
