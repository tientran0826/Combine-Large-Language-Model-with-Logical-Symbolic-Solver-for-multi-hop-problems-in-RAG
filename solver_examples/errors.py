import json
import pandas as pd


def check_errors(json_result_path):
  with open(json_result_path, 'r') as file:
    json_data = json.load(file)
  if len(json_data) == 0: return None
  df = pd.DataFrame(json_data)
  return f'Total correct syntax: {len(df[~df["has_error"].notnull()])}/{len(df)}'


if __name__ == '__main__':
  path = './outputs/self-refinement/refined_data_gpt-3.5-turbo.json'
  num = check_errors(path)
  print(num)
