import sys
import os
import en_core_web_sm
import json
import numpy as np
import random
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from typing import Any, List, Mapping, Tuple

class QuestionGenerator:

    def __init__(self) -> None:

        QG_PRETRAINED = "iarfmoose/t5-base-question-generator"
        self.ANSWER_TOKEN = "<answer>"
        self.CONTEXT_TOKEN = "<context>"
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=False)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

        self.qa_evaluator = QAEvaluator()

    # Generate the questions
    def generate_questions(self, question_count):

        print("Generating questions...\n")

        root_directory = './'  # Specify the root directory path here
        file_path = os.path.join(root_directory, 'ingest.txt')  # Path to the "ingest.txt" file
        
        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Generate questions using the content and question_count parameter
        qg_inputs, qg_answers = self.generate_q_inputs(content, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        # Validate
        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        # Evaluate
        if use_evaluator:
            print("Evaluating QA pairs...\n")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(
                generated_questions, qg_answers
            )
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)

            if num_questions:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores, num_questions
                )
            else:
                qa_list = self._get_ranked_qa_pairs(
                    generated_questions, qg_answers, scores
                )

        else:
            print("Skipping evaluation step.\n")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)

        return qa_list

if __name__ == '__main__':
# Read the command line arguments
arguments = sys.argv[1:]

# Parse the arguments
parameters = {}
for arg in arguments:
    key, value = arg.split('=')
    parameters[key] = value

# Get the value of the question_count parameter
question_count = int(parameters.get('question_count', '5'))

# Call the function to generate questions
q_generator = QuestionGenerator()
q_generator.generate_questions(question_count)
