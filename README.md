# environment setup

Install anaconda environment. 
Anaconda3-2023.03-1-Windows-x86_64 was used.

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
pip install sentencepiece
pip install protobuf==3.19.0

# Run code

Copy your text files to data directory.

Run the following cmd
    python ingest.py
    python generate_questions.py question_count=10
    python generate_answers.py