import argparse
import pandas as pd
import openai
import json
import os
from pydantic_core.core_schema import str_schema
from tqdm import tqdm
from models.utils import read_api_key
from nltk.inference import Prover9Command
from nltk.inference.prover9 import Expression
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel

import os
import gdown
import zipfile

# Define the file ID and output paths
file_id = '1iCKIGJOamE2Ki6TSajGiLwcO10_gVII_'
extract_path = './RAG_models'
output_path = os.path.join(extract_path, 'db_bge-large-en-v1.5.zip')

# Check if the extracted directory already exists
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
    # Construct the URL and download the file
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
    
    # Unzip the downloaded file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
else:
    print(f"Contents already extracted to {extract_path}.")
    
os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'
api_key = read_api_key('pyproject.toml')


class OpenAIModel:

    def __init__(self, api_key, model_name):
        self.model_name = model_name
        self.client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1", api_key=api_key
        ) if 'gpt' not in self.model_name else openai.OpenAI(api_key=api_key)

    def get_response(self, prompt, temperature=0.5):
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
            temperature=temperature,
        )
        return response.choices[0].message.content


class LogicProgram:

    def __init__(self, openai_client: OpenAIModel, prompt_path: str):
        self.prompt_path = prompt_path
        self.openai_client = openai_client

    def extract_logic_statements(self, text):
        lines = text.split('\n')
        premises, conclusion = [], ""
        in_premises, in_conclusion = False, False

        for line in lines:
            line = line.strip()
            if line.startswith("Premises:"):
                in_premises, in_conclusion = True, False
            elif line.startswith("Conclusion:"):
                in_premises, in_conclusion = False, True
            elif in_premises and line:
                premises.append(line)
            elif in_conclusion and line:
                conclusion = line

        return premises, conclusion

    def get_prover9_formula_from_llm(self, rules, goal):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read().replace("[Rules]",
                                         rules).replace("[Goals]", goal)
        return self.openai_client.get_response(prompt, temperature=0.5)

    def process_row(self, row):
        result = self.get_prover9_formula_from_llm(row['Law Context'],
                                                   row['Question'])
        premises, conclusion = self.extract_logic_statements(result)
        return {'raw': result, 'premises': premises, 'conclusion': conclusion}


class LogicInference:

    def prove_argument(self, data):
        try:
            conclusion = data['conclusion'].replace(".", "")
            premises = [p.replace(".", "") for p in data['premises']]
            goal, assumptions = Expression.fromstring(conclusion), [
                Expression.fromstring(a) for a in premises
            ]
            return Prover9Command(goal, assumptions=assumptions).prove(), None
        except Exception as e:
            return None, str(e)


class ErrorRefiner:

    def __init__(self,
                 openai_client: OpenAIModel,
                 prompt_path,
                 request_limit=5):
        self.prompt_path = prompt_path
        self.request_limit = request_limit
        self.openai_client = openai_client

    def prove(self, argument):
        try:
            goal, assumptions = argument
            g = Expression.fromstring(goal)
            alist = [Expression.fromstring(a) for a in assumptions]
            return None, Prover9Command(g, assumptions=alist).prove()
        except Exception as e:
            return e, None

    def extract_premises_and_conclusion(self, text):
        import re
        premises_pattern = re.compile(r"Premises:\s*\[([^\]]+)\]", re.DOTALL)
        conclusion_pattern = re.compile(r"Conclusion:\s*(.*)")

        premises_match = premises_pattern.search(text)
        premises = [
            p.strip().strip("'")
            for p in premises_match.group(1).strip().split(',')
        ] if premises_match else []

        conclusion_match = conclusion_pattern.search(text)
        conclusion = conclusion_match.group(
            1).strip() if conclusion_match else ""

        return premises, conclusion

    def process(self, data):
        premises, conclusion, error = data['premises'], data[
            'conclusion'], data['has_error']
        for _ in range(self.request_limit):
            with open(self.prompt_path, 'r') as f:
                refine_prompt = f.read().replace("[[PREMISES]]",
                                                 str(premises)).replace(
                                                     "[[CONCLUSION]]",
                                                     conclusion).replace(
                                                         "[[ERROR MESSAGES]]",
                                                         str(error))
            response = self.openai_client.get_response(refine_prompt,
                                                       temperature=1)

            premises, conclusion = self.extract_premises_and_conclusion(
                response)
            argument = (conclusion.replace(".", ""),
                        [p.replace(".", "") for p in premises])
            error, is_proof = self.prove(argument)

            if is_proof:
                data.update({
                    'premises': premises,
                    'conclusion': conclusion,
                    'prover9_result': str(is_proof),
                    'has_error': None
                })
                break
            else:
                data["has_error"] = error

        return data


class LogicReasoning:

    def __init__(self, openai_client, logic_program_data, refined_data,
                 prompt_path):
        self.openai_client = openai_client
        self.refined_data = refined_data
        self.raw = logic_program_data['raw']
        self.prompt_path = prompt_path

    def get_answer(self, raw, premises, conclusion, prover9_result):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read().replace("[[RAW]]", raw).replace(
                "[[PREMISES]]",
                str(premises)).replace("[[CONCLUSION]]", conclusion).replace(
                    "[[PROVER9_RESULT]]", prover9_result)
        response = self.openai_client.get_response(prompt, temperature=0.5)
        return response.replace("RESULT EXPLAIN:",
                                "").replace("Here is the explanation:", "")


class LogicPipeline:

    def __init__(self, openai_client: OpenAIModel, parser_prompt_path: str,
                 self_refine_prompt_path: str,
                 logic_reasoning_prompt_path: str):
        self.openai_client = openai_client
        self.logic_program = LogicProgram(openai_client, parser_prompt_path)
        self.logic_inference = LogicInference()
        self.refine_errors = ErrorRefiner(openai_client,
                                          self_refine_prompt_path)
        self.logic_reasoning_prompt_path = logic_reasoning_prompt_path

    def process_input(self, input_data):
        logic_program_output = self.logic_program.process_row(input_data)
        logic_inference_output, error = self.logic_inference.prove_argument(
            logic_program_output)
        output_data = {
            "conclusion": logic_program_output['conclusion'],
            "premises": logic_program_output['premises'],
            "prover9_result": str(logic_inference_output),
            "has_error": error
        }

        if error is not None:
            output_data = self.refine_errors.process(output_data)

        logic_reasoning = LogicReasoning(self.openai_client,
                                         logic_program_output, output_data,
                                         self.logic_reasoning_prompt_path)
        answer = logic_reasoning.get_answer(logic_program_output['raw'],
                                            output_data['premises'],
                                            output_data['conclusion'],
                                            output_data['prover9_result'])
        return answer
    
class LawCleaning:
    def __init__(self, openai_client, raw_law, prompt_path):
        self.openai_client = openai_client
        self.raw_law = raw_law
        self.prompt_path = prompt_path

    def get_answer(self):
        with open(self.prompt_path, 'r') as file:
            prompt = file.read()
        prompt = prompt.replace("[[LAW CONTEXT]]", self.raw_law)

        response = self.openai_client.get_response(
            prompt=prompt,
            temperature=0.5,
        )

        return response.replace("Here is the cleaned context:", "").replace("Here is the Clean Context:", "").replace("Clean Context:", "")

class EmbeddingFunction_custom(EmbeddingFunction):

    def __init__(self, model_name):

        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, convert_to_tensor=True).tolist()


def get_chroma_db(chroma_path, name_collection, model_embedding):
    chroma_client = PersistentClient(path=chroma_path)

    db = chroma_client.get_or_create_collection(
        name=name_collection,
        embedding_function=EmbeddingFunction_custom(model_embedding))
    return db


def query_db(query_text, db, k=30):
    return db.query(query_texts=[query_text], n_results=k)

def query(
    question,
    db,
    reranker,
    num_retrieved_docs=30,
    num_docs_final=5,
):
    # Gather documents with retriever
    relevant_docs = query_db(question, db,
                             k=num_retrieved_docs)['documents'][0]
    # Optionally rerank results
    if reranker:
        relevant_docs = reranker.rerank(question,
                                        relevant_docs,
                                        k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]
    return relevant_docs


if __name__ == '__main__':
    model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
    logic_program_path = './models/prompts/prover9-parsrer.txt'
    self_refine_path = './models/prompts/self-correct.txt'
    logic_reasoning_prompt_path = './models/prompts/explain-result.txt'
    cleaning_prompt_path = './models/prompts/cleaning-data.txt'
    db_path = f'{extract_path}/db_bge-large-en-v1.5'
    name_collection = 'db_law'
    version = "BAAI/bge-large-en-v1.5"
    reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0",verbose = 0)
    db = get_chroma_db(db_path, name_collection, version)
    ques = 'Can I drive a car after consuming alcohol? If there is a fine, what is the penalty for this offense?'
    
    relevant_docs = query(ques, db, reranker)
    law_context = relevant_docs[0]
    open_ai = OpenAIModel(api_key, model_name)
    law_cleaning = LawCleaning(open_ai, law_context, cleaning_prompt_path).get_answer()
    input_data = {
        "Question": ques,
        "Law Context": [law_cleaning]
    }
    
    logic_pipeline = LogicPipeline(open_ai, logic_program_path,
                                   self_refine_path,
                                   logic_reasoning_prompt_path)
    for rag_law in input_data['Law Context']:
        row_input_data = {
            'Question': input_data['Question'],
            'Law Context': rag_law
        }
        answer = logic_pipeline.process_input(row_input_data)
        print(row_input_data['Question'])
        print(law_context)
        print(answer)
