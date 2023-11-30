from PyPDF2 import PdfReader
import pandas as pd
import re

# Function to fix Thai spacing, targeting only spaces within words
def fix_thai_spacing_advanced(text):
    # Regular expression to identify Thai characters
    thai_char = r'[\u0E00-\u0E7F]'

    # Regular expression to identify spaces within Thai words
    # Matches a Thai character, followed by a space, followed by a Thai character
    # space_within_word_pattern = r'(?<=' + thai_char + r')\s+(?=' + thai_char + r')'
    space_within_word_pattern = r'(?<=' + thai_char + r')[^\S\n]+(?=' + thai_char + r')'

    # Remove spaces within Thai words
    fixed_text = re.sub(space_within_word_pattern, '', text)

    return fixed_text

# Function to read and fix PDF text
def read_and_fix_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    fixed_text = ''

    for page in reader.pages:
        text = page.extract_text()
        if text:
            fixed_text += fix_thai_spacing_advanced(text)
            # fixed_text += text

    return fixed_text


# Main processing function
def process_pdf(pdf_path):
    fixed_pdf_text = read_and_fix_pdf(pdf_path)
    # Regex pattern to identify 'หมวดที่' and its description
    pattern = r'(หมวดที่  \d+.*?)\n(.*?)\n'
    matches = re.finditer(pattern, fixed_pdf_text, re.DOTALL)
    data = []
    for match in matches:
        section_number = match.group(1)
        section_description = match.group(2).strip()
        
        # Find the start and end indices of the section text
        start_index = match.end()
        if match.end() < len(fixed_pdf_text):
            next_section = re.search(pattern, fixed_pdf_text[match.end():], re.DOTALL)
            if next_section:
                end_index = match.end() + next_section.start()
            else:
                end_index = len(fixed_pdf_text)
        else:
            end_index = len(fixed_pdf_text)

        section_text = fixed_pdf_text[start_index:end_index]
        data.append([section_number, section_description, section_text])

    return pd.DataFrame(data, columns=['Section', 'Description', 'Text'])

# Path to your PDF file
pdf_path = 'assets/main/Work regulations TIP 21-10-2021.pdf'

# Define chunk size and overlap
# chunk_size = 1000  # Example chunk size
# overlap = 200  # Example overlap

# Process the PDF and save to CSV
df = process_pdf(pdf_path)
# Hardcode fix
df.iloc[11].Text = df.iloc[11].Text + df.iloc[12].Section + df.iloc[12].Description + df.iloc[12].Text

df.to_csv('extracted_data/main_doc_metadata.csv', index=False)