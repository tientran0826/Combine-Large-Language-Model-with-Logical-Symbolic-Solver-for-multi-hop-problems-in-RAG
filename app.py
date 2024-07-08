from flask import Flask, request, jsonify, render_template
from logic_pipeline import OpenAIModel, LogicPipeline, LawCleaning, get_chroma_db, RAGPretrainedModel, query

import os
from models.utils import read_api_key

app = Flask(__name__)

# Initialize components needed for processing
model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
logic_program_path = './models/prompts/prover9-parsrer.txt'
self_refine_path = './models/prompts/self-correct.txt'
logic_reasoning_prompt_path = './models/prompts/explain-result.txt'
cleaning_prompt_path = './models/prompts/cleaning-data.txt'
extract_path = './RAG_models'
db_path = f'{extract_path}/db_bge-large-en-v1.5'
name_collection = 'db_law'
version = "BAAI/bge-large-en-v1.5"


os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'
api_key = read_api_key('pyproject.toml')
reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", verbose=0)
open_ai = OpenAIModel(api_key, model_name)
db = get_chroma_db(db_path, name_collection, version)

# Initialize logic pipeline
logic_pipeline = LogicPipeline(open_ai, logic_program_path, self_refine_path, logic_reasoning_prompt_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        
        # Query relevant documents
        relevant_docs = query(question, db, reranker)
        law_context = relevant_docs[0]

        # Clean law context
        law_cleaning = LawCleaning(open_ai, law_context, cleaning_prompt_path).get_answer()

        # Prepare input data for logic pipeline
        input_data = {
            "Question": question,
            "Law Context": [law_cleaning]
        }

        # Process input through logic pipeline
        results = []
        for rag_law in input_data['Law Context']:
            row_input_data = {
                'Question': input_data['Question'],
                'Law Context': rag_law
            }
            answer = logic_pipeline.process_input(row_input_data)
            results.append({
                'Question': row_input_data['Question'],
                'Law Context': law_cleaning,
                'Answer': answer
            })

        return render_template('index.html', results=results, question=question)

    return render_template('index.html')

@app.route('/project-detail', methods=['GET'])
def project_detail():
    return render_template('project_detail.html')

if __name__ == '__main__':
    app.run(debug=True)
