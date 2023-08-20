from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.document_loaders import TextLoader
loader = TextLoader('ingest.txt', encoding='utf-8')
documents = loader.load()

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-248M", model_kwargs={"temperature":0.2, "max_length":4096})
# llm=HuggingFaceHub(repo_id="gaurangak/text-davinci-003", model_kwargs={"temperature":0.2, "max_length":500})
# llm=HuggingFaceHub(repo_id="TheBloke/Llama-2-7B-Chat-GGML", model_kwargs={"temperature":0.2, "max_length":4096})
chain = load_qa_chain(llm, chain_type="stuff")

# Read the questions from the file
questions_file = "questions.txt"
with open(questions_file, "r", encoding='utf-8') as file:
    questions = file.readlines()
questions = [q.strip() for q in questions]

# Answer the questions
answers = []
for question in questions:
    docs = db.similarity_search(question)
    answer = chain.run(input_documents=docs, question=question)
    answers.append(answer)

# Print the answers
for question, answer in zip(questions, answers):
    print("Question:", question)
    print("Answer:", answer)
    print()