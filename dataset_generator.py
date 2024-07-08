import json
import random
import re
def is_true_false_question(question_text):
    """
    Determines if the given question is a True/False question.
    Checks if the question starts with True/False keywords.
    """
    true_false_keywords = ['is', 'are', 'does', 'do', 'did', 'can', 'should', 'has', 'have', 'was', 'were', 'will', 'could']
    return any(question_text.lower().strip().startswith(keyword) for keyword in true_false_keywords)

def remove_conclusion(answer_text):
    """
    Removes the conclusion from the answer text based on specific keywords such as
    'Therefore', 'So', 'Thus', etc. Assumes the conclusion starts with one of these keywords.
    """
    conclusion_keywords = [
        'Therefore', 'So', 'Thus', 'Hence', 'Consequently', 'As a result', 'Accordingly', 
        'For this reason', 'Based on the above conditions', 'Given the above', 'In conclusion', 
        'To conclude', 'Summarizing', 'On this basis', 'It follows that', 'This implies that', 
        'In summary'
    ]
    pattern = r'(?<!\w)(' + '|'.join(conclusion_keywords) + r')\b.*'
    truncated_answer = re.split(pattern, answer_text)[0]
    return truncated_answer.strip()

def remove_question_from_answer(question_text, answer_text):
    """
    Removes the question text from the answer text if it is repeated within the answer.
    """
    question_text = question_text.strip()
    if question_text in answer_text:
        return answer_text.replace(question_text, '').strip()
    return answer_text

def process_qa(qa):
    """
    Processes the QA dictionary to remove the question text from the answer if repeated,
    remove the conclusion from the answer, and set the processed answer as the "Law Context".
    """
    question_text = qa.get("question_text", "")
    answer_text = qa.get("answer_text", "")
    
    if question_text and answer_text:
        answer_text = remove_question_from_answer(question_text, answer_text)
        answer_text = remove_conclusion(answer_text)
    
    qa["answer_text"] = answer_text
    return qa

def create_random_true_false_dataset(input_file, output_file, num_samples=1000):
    """
    Creates a new dataset with a random selection of True/False questions, processes the answers,
    and writes the processed records to an output JSONL file.
    """
    # Read all QA pairs from the input file
    true_false_qa_pairs = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            qa = json.loads(line.strip())
            if is_true_false_question(qa.get("question_text", "")):
                true_false_qa_pairs.append(qa)
    
    # Select num_samples True/False questions randomly
    selected_qa_pairs = random.sample(true_false_qa_pairs, min(num_samples, len(true_false_qa_pairs)))
    
    # Process and write the selected QA pairs to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for qa in selected_qa_pairs:
            processed_qa = process_qa(qa)
            # Extract desired columns and rename them
            id_ = qa.get("_id", "")
            question = qa.get("question_text", "")
            law_context = processed_qa.get("answer_text", "")
            # Write to the output file
            json.dump({"Id": id_, "Question": question, "Law Context": law_context}, outfile)
            outfile.write('\n')  # Add newline for JSONL format
# Example usage
input_file = './data/en_legal_question_answer_dataset.jsonl'
output_file = './data/fol_dataset.jsonl'
create_random_true_false_dataset(input_file, output_file, num_samples=500)
