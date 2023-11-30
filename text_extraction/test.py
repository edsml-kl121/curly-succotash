import re

def fix_thai_spacing_advanced(text):
    # Remove all spaces
    text_no_spaces = re.sub(r'\s+', '', text)

    # Regular expression to identify places where spaces should be reinserted
    # e.g., before numbers, punctuation, or Latin characters
    space_insertion_pattern = r'(?<=[\u0E00-\u0E7F])(?=[0-9A-Za-z,.!?])'

    # Insert space at the identified positions
    fixed_text = re.sub(space_insertion_pattern, ' ', text_no_spaces)

    return fixed_text

# Your text
text = """
ส ิ ่ ง จ ู ง ใ จ ใ น ก า ร ท  า ง า น 
... (rest of the text)
"""

# Apply the fix
fixed_text = fix_thai_spacing_advanced(text)
print(fixed_text)