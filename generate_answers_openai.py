from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
# from llama_index.node_parser import SimpleNodeParser
from langchain import OpenAI

import os
import config
import openai
# set OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
# Set your OpenAI API key
openai.api_key = config.OPENAI_API_KEY

# get parameter
# data = request.get_json()
# query_result = data.get('queryResult')
# query = query_result.get('queryText')

# load training data from directory {root}/data/*
documents = SimpleDirectoryReader('data').load_data()

# parse the document into nodes
# parser = SimpleNodeParser()
# nodes = parser.get_nodes_from_documents(documents)

# index construction
# index = GPTVectorStoreIndex.from_documents(documents)
# index = GPTVectorStoreIndex(nodes)

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 0.1
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

index = GPTVectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# save the index for future use - save ./storage as default
index.storage_context.persist()

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="storage")

# load index
index = load_index_from_storage(
    storage_context, service_context=service_context
)

query_engine = index.as_query_engine()

# Read the questions from the file
questions_file = "questions.txt"
with open(questions_file, "r", encoding='utf-8') as file:
    questions = file.readlines()
questions = [q.strip() for q in questions]

# Answer the questions
answers = []
for question in questions:
    answer = query_engine.query(question)
    answers.append(answer)

# Print the answers
for question, answer in zip(questions, answers):
    print("Question:", question)
    print("Answer:", answer)
    print()