import os

root_directory = './'  # Specify the root directory path here
data_directory = os.path.join(root_directory, 'data')  # Path to the "data" directory
output_file_path = os.path.join(root_directory, 'ingest.txt')  # Path to the output file

with open(output_file_path, 'w') as output_file:
    # Iterate over all files in the "data" directory
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                output_file.write(text)
                output_file.write('\n')  # Add a newline after each file's content
