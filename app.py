import logging
import os
import pickle
import tempfile
import textwrap
import sys

import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import \
    GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
# from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings,
                                  HuggingFaceEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma, Milvus
from pymilvus import connections, Collection
from langchain.prompts import PromptTemplate
from PIL import Image
from tools.translation import translate_large_text, translate_to_thai
import requests
import time
import fitz
import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer, models
from langChainInterface import LangChainInterface
# from tools.backend_helper import read_pdfs, listing_docs
from tools.frontend_helper import initialize_db_client, get_db_results, \
    open_pdf, get_model

load_dotenv()
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
environment = os.getenv("ENVIRONMENT", "local")

script_dir = os.path.dirname(__file__)
# Start app mode
if environment != "local":
    # Get current time and format it for the filename (e.g., 'application_28-Nov-23_04-26-12.log')
    current_time = datetime.datetime.now()
    # Format the timestamp (for example, '20231128-150245' for 28th Nov 2023, 3:02:45 PM)
    timestamp = current_time.strftime("%Y%m%d-%H%M%S")

    # Construct the full path to the image

    # Generate the filename with the timestamp suffix
    filename = f"logs/logs-{timestamp}.log"
    logs_path = os.path.join(script_dir, filename)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s',
                        handlers=[logging.FileHandler(logs_path),
                                  logging.StreamHandler(sys.stdout)],
                        datefmt='%d-%b-%y %H:%M:%S')
else:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s',
                        handlers=[logging.FileHandler("debug.log"),
                                  logging.StreamHandler(sys.stdout)],
                        datefmt='%d-%b-%y %H:%M:%S')

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.image("assets/images/")
st.header("Rag 💬")
# chunk_size=1500
# chunk_overlap = 200


if api_key is None or ibm_cloud_url is None or project_id is None:
    print(
        "Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

def format_text(i):
    # Chunk 1
    translated_documents = translate_to_thai(docs[i].page_content, True)
    currentdoc = docs[i]
    full_text = f"""Content:
    {translated_documents}

Source: {currentdoc.sources}, Page {currentdoc.pagesno + 1}
    """
    return full_text


# show user input
if user_question := st.text_input(
    "ถามคำถามเกี่ยวกับโปรโมชั่นของคุณ:"
):
    logging.info(user_question)
    model_response_placeholder = st.empty()
    # Placeholder for dynamic text and spinner
    with st.spinner('Initialization...'):
        # model_response_placeholder.text_area(label="Model Response", value="generating...", height=300)
        translated_user_input = translate_large_text(user_question,
                                                    translate_to_thai, False)
        model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                        max_seq_length=384)

    with st.spinner('Retrieving relevent chunk...'):
        docs = get_db_results(translated_user_input, model,
                            collection_name="dhipaya_hr_policy",
                            n_results=4)

    with st.spinner('Model generating response...'):
        for i in range(4):
            chunk_label = f"ข้อมูลเกี่ยวข้องในเอกสารอันดับ {i + 1}"
            source_button_label = f"View source {i + 1}"

            # Display the text area for the chunk
            st.text_area(label=chunk_label, value=format_text(i), height=100)

            # Create a button to view the source; if clicked, open the PDF
            if st.button(source_button_label):
                open_pdf(docs[i].sources, docs[i].pagesno)

        print('docs' + "*" * 5)
        print(docs)
        print("*" * 5)

        params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MIN_NEW_TOKENS: 10,
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.TEMPERATURE: 0.0,
            GenParams.STOP_SEQUENCES: ['END_KEY'],
            # GenParams.TOP_K: 100,
            # GenParams.TOP_P: 1,
            GenParams.REPETITION_PENALTY: 1
        }
        model_llm = LangChainInterface(model=ModelTypes.LLAMA_2_70B_CHAT.value,
                                    credentials=creds, params=params,
                                    project_id=project_id)

        knowledge_based_template = (
            open("assets/llama2-prompt-template-rag.txt",
                encoding="utf8").read().format(
            )
        )
        custom_prompt = PromptTemplate(template=knowledge_based_template,
                                    input_variables=["context", "question"])
        chunks_combined = ""
        for i in range(len(docs)):
            chunks_combined += f"chunk {i} \n" + docs[i].new_translated_docs + "\n"

        # logging.info("start translate chunk combined")
        # translated_chunks_combined = translate_large_text(chunks_combined,
        #                                                   translate_to_thai, False)
        # logging.info("end translate chunk combined")

        formated_prompt = custom_prompt.format(question=translated_user_input,
                                            context=chunks_combined)
        logging.info(formated_prompt)

        logging.info("start generate")
        response = model_llm(custom_prompt.format(question=translated_user_input,
                                                context=formated_prompt))
        logging.info("end generate")

        start = time.time()
        response = model_llm(custom_prompt.format(question=translated_user_input, context=formated_prompt))
        end = time.time()
        
        logging.info("model generation: ", end - start)

        start = time.time()
        # Response
        logging.info("start translate response")
        translated_response = translate_to_thai(response, True)
        logging.info("end translate response")

    # translated_response = translated_response.replace("<|endoftext|>", "")
    # st.text_area(label="Model Response", value=translated_response, height=300)
    model_response_placeholder.text_area(label="คำตอบของ watsonx ai", value=translated_response, height=300)

    st.write()
