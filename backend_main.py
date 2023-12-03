from dotenv import load_dotenv
import os
# from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings,
                                  SentenceTransformerEmbeddings)
from langchain.vectorstores import FAISS, Chroma, Milvus
from pymilvus import connections, DataType
import requests

# importing Custom package
from tools.translation import translate_large_text, translate_to_thai
from tools.backend_helper import read_pdfs, listing_docs, manage_collection
from tools.frontend_helper import get_model

import pandas as pd

load_dotenv()
milvus_host = os.getenv("MILVUS_HOST", None)
milvus_port = os.getenv("MILVUS_PORT", None)
connections.connect(host=milvus_host,port=milvus_port)

print("Chunking docs")
base_folder_path = 'assets/extracted_data'
# translated_docs = read_pdfs(listing_docs(base_folder_path))

def get_data(file_path='assets/extracted_data'):
    incidence_df = pd.read_csv(file_path)
    incidence_df = incidence_df.astype(str)

    # Get the column names from the DataFrame
    column_names = incidence_df.columns

    # Initialize an empty dictionary to store lists for each column
    data_dict = {}

    # Loop through each column and translate its values
    for column in column_names:
        print(column)
        # Translate column values to English
        english_values = [translate_large_text(value, translate_to_thai, False, 250) for value in incidence_df[column]]
        print("done translating")

        # Translate column values to Thai
        thai_values = incidence_df[column].tolist()
        
        # Create a dictionary with English and Thai versions
        data_dict[column + '_en'] = english_values
        data_dict[column + '_th'] = thai_values
    return data_dict

data_dict_main_doc_chunks = get_data('text_extraction/extracted_data/main_doc_chunks.csv')

data_dict_main_doc_chunks["text_to_encode"] = [
    f"section: {section}\n description: {description}\n chunk: {chunk}"
    for section, description, chunk in zip(data_dict_main_doc_chunks["Section_en"], data_dict_main_doc_chunks["Description_en"], data_dict_main_doc_chunks["chunk_en"])
]
print(data_dict_main_doc_chunks.keys())
print(data_dict_main_doc_chunks)


collection_name = "dhipaya_main_doc"
print(collection_name)
schema = \
{
    "embeddings": DataType.FLOAT_VECTOR, 
    "text_to_encode": DataType.VARCHAR, 
    "Section": DataType.VARCHAR, 
    "Description": DataType.VARCHAR,
    "chunk": DataType.VARCHAR
}
collection = manage_collection(collection_name, schema)
print("initialized new collection")


print("ingesting into Milvus - start")
model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2", max_seq_length=384)
embeds = [list(embed) for embed in model.encode(data_dict_main_doc_chunks["text_to_encode"])]
collection.insert([embeds, data_dict_main_doc_chunks["text_to_encode"], data_dict_main_doc_chunks["Section_th"], data_dict_main_doc_chunks["Description_th"], data_dict_main_doc_chunks["chunk_th"]])
collection.create_index(field_name="embeddings",\
                        index_params={"metric_type":"IP","index_type":"IVF_FLAT","params":{"nlist":16384}})

print("ingesting into Milvus - completed")