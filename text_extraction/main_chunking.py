import pandas as pd
import re

# Function to chunk text with overlap
def chunk_text(text, chunk_size, overlap):
    pattern = r'-\d+-'
    text_cleaned = re.sub(pattern, '', text)
    return [text_cleaned[i:i+chunk_size] for i in range(0, len(text_cleaned), chunk_size - overlap)]

def extract_sections_2(text):
    # Extract Section 1
    start_part_1 = text.find("ข้อ  1 .")
    end_part_1 = text.find("-2-", start_part_1)
    part_1 = text[start_part_1:end_part_1]

    # Extract Introduction and Section 2
    end_introduction = start_part_1
    introduction = text[:end_introduction]

    start_part_2 = text.find("ข้อ  2 .")
    end_part_2 = text.find("-6-", start_part_2)
    part_2 = text[start_part_2:end_part_2]

    return introduction, part_1, part_2

def extract_sections_7(text, num_parts):
    parts = []

    for i in range(1, num_parts + 1):
        start_marker = f"ข้อ  {i} ."
        end_marker = f"ข้อ  {i+1} ." if i < num_parts else None  # The end marker for the last part is None

        start = text.find(start_marker)
        end = text.find(end_marker, start) if end_marker else len(text)

        part = text[start:end].strip()  # Remove any leading/trailing whitespace
        parts.append(part)

    return parts
def process_section_7_part_3(text):
    pattern = r'(ตักเตือนด้วยวาจา)(.*?)(ตักเตือนเป็นหนังสือ)(.*?)(ตัดเงินรางวัลพิเศษประจาปี)(.*?)(เลิกจ้าง)(.*)'

    matches = re.search(pattern, text, re.DOTALL)
    action_authorer = ""
    if matches:
        for i in range(1, 8, 2):
            action = matches.group(i).strip()
            authority = matches.group(i + 1).strip()
            action_authorer += f"\n action to authorer pair: \n ระดับโทษทางวินัย: {action} - ผู้มีอ านาจหน้าที่ลงโทษ: {authority}\n"
            # print(f"Action: {action} - Authorizor: {authority}\n")
    intro_sentence = """ข้อ  3 . อานาจหน้าที่ในการลงโทษทางวินัย  
            ให้ผู้บังคับบัญชาตามสายงานมีสิทธิและอานาจหน้าที่ในการลงโทษทางวินัยดังนี้"""
    # print(intro_sentence + action_authorer)
    return intro_sentence + action_authorer

metadata = pd.read_csv("extracted_data/main_doc_metadata.csv")

# print(metadata)
titles_simple_format = [
  "หมวดที่  2 ",
  "หมวดที่  3 ",
  "หมวดที่  4 ",
  "หมวดที่  5 ",
  "หมวดที่  6 ",
  "หมวดที่  8  ",
  "หมวดที่  9 ",
  "หมวดที่  10 ",
  "หมวดที่  1 1  "
]

titles_complex_format = [
  "หมวดที่  1 ",
  "หมวดที่  7 ",
]

data = []
for index, row in metadata.iterrows():
    if row["Section"] in titles_simple_format:
        print(row['Section'])
        chunks = chunk_text(row['Text'], 500, 100)
        print(len(chunks))
        for chunk in chunks:
            data.append([row['Section'], row['Description'], chunk])




# data = []
for index, row in metadata.iterrows():
    if row["Section"] in titles_complex_format:
        if row["Section"] == titles_complex_format[0]:
            introduction, part_1, part_2 = extract_sections_2(row['Text'])
            chunks = chunk_text(introduction + part_2, 500, 100)
            for chunk in chunks:
                data.append([row['Section'], row['Description'], chunk])
        elif row["Section"] == titles_complex_format[1]:
            part_1, part_2, part_3, part_4, part_5 = extract_sections_7(row['Text'], 5)
            revised_pattern_with_space = r'(\d \.\d\.\d{1,2} ?\d{0,2})\s*(.*?)ตั้งแต่(.*?)จนถึง(.*?)(?=\d \.\d\.\d{1,2} ?\d{0,2}|$)'
            replacement = r'\1\nวินัย: \2\nระดับโทษ: ตั้งแต่\3จนถึง\4\n'
            part_2_new = re.sub(revised_pattern_with_space, replacement, part_2, flags=re.DOTALL)
            chunks = chunk_text(part_1 + "\n" + part_2_new + "\n" + process_section_7_part_3(part_3) + "\n" + part_4 +"\n" +  part_5, 500, 100)
            
            for chunk in chunks:
                data.append([row['Section'], row['Description'], chunk])

final_df = pd.DataFrame(data, columns=['Section', 'Description', "chunk"])

final_df.to_csv('extracted_data/main_doc_chunks.csv', index=False)
            