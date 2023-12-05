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


# Function to extract dates using a regular expression
def extract_dates(text):
    pattern = r"ตั้งแต่วันที่ (\d{1,2} \S+ \d{4})|สั่ง ณ วันที่ (\d{1,2} \S+ \d{4})"
    matches = re.findall(pattern, text)
    return [date for match in matches for date in match if date]

# Function to extract commands with 'คำสั่งที่' followed by text until 'เรื่อง'
def extract_commands(text):
    pattern = r"คำสั่งที่ (.*?)เรื่อง"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

# Function to extract subjects with 'เรื่อง' followed by text until a newline
def extract_subjects(text):
    pattern = r"เรื่อง (.*?)\n"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]




def read_pdfs(file_names):

    data = []
    for file in file_names:
        file_path = f"assets/changes_txt/{file}"
        print("current file: ", file)
        # Using with statement for better resource management
        with open(file_path, 'r') as file_reader:
            # Reading the content of the file
            output = file_reader.read()
            # Extract and print the dates, commands, and subjects
            dates = extract_dates(output)
            print("Dates:", dates)

            commands = extract_commands(output)
            print("Commands:", commands)

            subjects = extract_subjects(output)
            print("Subjects:", subjects)

        data.append([file, commands[0], dates[0], dates[1], subjects[0], output])
    return pd.DataFrame(data, columns=['file_name', "command_no", "effective_date", "date_ordered", "subject", 'Description'])

df = read_pdfs(listing_docs(folder_path='assets/changes_txt'))

df.to_csv('extracted_data/changes_policies.csv', index=False)
