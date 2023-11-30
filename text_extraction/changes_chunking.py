import pandas as pd
import re
import os

def listing_docs(folder_path='assets/changes_txt'):
    ## Specify the relative path to the 'pdfs' folder from your script's location

    # Get a list of files in the 'pdfs' directory
    files_in_directory = os.listdir(folder_path)

    # Filter out directories, leaving only files
    file_names = [file for file in files_in_directory if os.path.isfile(os.path.join(folder_path, file))]

    print(file_names)
    return file_names


def read_pdfs(file_names):

    data = []
    for file in file_names:
        file_path = f"assets/changes_txt/{file}"
        print("current file: ", file)
        # Using with statement for better resource management
        with open(file_path, 'r') as file_reader:
            # Reading the content of the file
            output = file_reader.read()
        data.append([file, output])
    return pd.DataFrame(data, columns=['file_name', 'Description'])

df = read_pdfs(listing_docs(folder_path='assets/changes_txt'))

df.to_csv('extracted_data/changes_policies.csv', index=False)
