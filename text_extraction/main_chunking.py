import pandas as pd

# Function to chunk text with overlap
def chunk_text(text, chunk_size, overlap):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

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
final_df = pd.DataFrame(data, columns=['Section', 'Description', "chunk"])

final_df.to_csv('extracted_data/main_doc_chunks.csv', index=False)

for index, row in metadata.iterrows():
    if row["Section"] in titles_complex_format:
        print(row['Section'])