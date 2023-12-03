from dotenv import load_dotenv
import os
# from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings,
                                  HuggingFaceEmbeddings)
from langchain.vectorstores import FAISS, Chroma, Milvus
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer, models
import fitz
from PIL import Image
milvus_host = os.getenv("MILVUS_HOST", None)
milvus_port = os.getenv("MILVUS_PORT", None)

def initialize_db_client(host=milvus_host, port=milvus_port):
    """
    Initializes and returns a chromadb client.

    Parameters:
    - host (str): The host for the chromadb service. Default is 'localhost'.
    - port (int): The port for the chromadb service. Default is 8000.

    Returns:
    - chromadb.HttpClient: An initialized chromadb client.
    """
    return connections.connect(host=host,port=port)


def get_db_results_main(query_text, model, collection_name="collection", n_results=4):
    """
    Queries the given collection in the database with the provided query text and returns results.

    Parameters:
    - query_text (str): The text to be queried.
    - collection_name (str): The name of the collection to query. Default is "law_topics".
    - n_results (int): Number of results to fetch. Default is 1.

    Returns:
    - dict: Query results.
    """

    client = initialize_db_client()
    query_encode = [list(i) for i in model.encode([query_text])]
    collection = Collection(collection_name)
    collection.load()
    documents = collection.search(data=query_encode, anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text_to_encode", "Section", "Description", "chunk"], limit=4)
    return documents[0]

def get_db_results_terms(query_text, model, collection_name="collection", n_results=4):
    """
    Queries the given collection in the database with the provided query text and returns results.

    Parameters:
    - query_text (str): The text to be queried.
    - collection_name (str): The name of the collection to query. Default is "law_topics".
    - n_results (int): Number of results to fetch. Default is 1.

    Returns:
    - dict: Query results.
    """

    client = initialize_db_client()
    query_encode = [list(i) for i in model.encode([query_text])]
    collection = Collection(collection_name)
    collection.load()
    documents = collection.search(data=query_encode, anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text_to_encode", "Term_th", "Meaning_th"], limit=4)
    return documents[0]

def get_db_results_changes(query_text, model, collection_name="collection", n_results=4):
    """
    Queries the given collection in the database with the provided query text and returns results.

    Parameters:
    - query_text (str): The text to be queried.
    - collection_name (str): The name of the collection to query. Default is "law_topics".
    - n_results (int): Number of results to fetch. Default is 1.

    Returns:
    - dict: Query results.
    """

    client = initialize_db_client()
    query_encode = [list(i) for i in model.encode([query_text])]
    collection = Collection(collection_name)
    collection.load()
    documents = collection.search(data=query_encode, anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text_to_encode", "file_name", "Description"], limit=4)
    return documents[0]



def get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768, condition=True):
    if condition:
        # model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        # model_name = "hkunlp/instructor-large"
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode='cls') # We use a [CLS] token as representation
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def open_pdf(pdf_path, page_num):
    # Opening the PDF file and creating a handle for it
    file_handle = fitz.open(pdf_path)

    # The page no. denoted by the index would be loaded
    page = file_handle[page_num]

    # Set the desired DPI (e.g., 200)
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

    # Obtaining the pixelmap of the page
    page_img = page.get_pixmap(matrix=mat)

    # Saving the pixelmap into a png image file
    page_img.save('PDF_page_high_res.png')

    # Reading the PNG image file using pillow
    img = Image.open('PDF_page_high_res.png')

    # Displaying the png image file using an image viewer
    img.show()
