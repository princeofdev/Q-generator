from transformers import pipeline

# Load the question answering pipeline
nlp = pipeline("question-answering")

# Read the questions from the file
questions_file = "questions.txt"  # Path to the file containing the questions
with open(questions_file, "r") as file:
    questions = file.readlines()
questions = [q.strip() for q in questions]

# Load the context text
context_file = "pairs.txt"  # Path to the file containing the context
with open(context_file, "r") as file:
    context = file.read()

# Answer the questions
answers = []
for question in questions:
    answer = nlp(question=question, context=context)
    answers.append(answer)

# Print the answers
for question, answer in zip(questions, answers):
    print("Question:", question)
    print("Answer:", answer["answer"])
    print("Confidence score:", answer["score"])
    print()
