import pandas as pd
import json
import openai
import toml
import os


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
    return model_name


api_key = read_api_key('pyproject.toml', openai=True)
# client = openai.OpenAI(base_url="https://api.endpoints.anyscale.com/v1",
#                        api_key=api_key)
client = openai.OpenAI(api_key=api_key)


naive_prompt_path = './baselines/prompts/naive.txt'
cot_prompt_path = './baselines/prompts/cot.txt'
with open(naive_prompt_path, 'r') as file:
    naive_prompt = file.read()


def extract_logic_statements(text):
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if line:
            for statement in ["TRUE", "FALSE", "UNCERTAIN"]:
                if statement in line:
                    return statement
    return None


def get_naive_from_llm(model_name, prompt, rules, question):
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
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def process_row(row, model_name):
    raw_result = get_naive_from_llm(model_name, naive_prompt,
                                    row['Law Context'], row['Question'])
    result = extract_logic_statements(raw_result)
    return pd.Series({'Naive_result': result})


# Read the JSON file into a DataFrame
df = pd.read_json("./data/dataset.json")

# List of models to apply
model_names = [
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "mistralai/Mixtral-8x22B-Instruct-v0.1",
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    "gpt-3.5-turbo"
]

for model_name in model_names:
    model_name_path = get_model_name_path(model_name)
    results = df.apply(process_row, axis=1, args=(model_name, ))
    df['naive_result'] = results['Naive_result']
    result_df = df[['Id', 'naive_result']]
    output_path = os.path.join('./baselines/outputs/naive',
                               f'result_{model_name_path}.json')
    result_df.to_json(output_path, orient='records', lines=True)
