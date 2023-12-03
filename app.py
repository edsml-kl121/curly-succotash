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
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.metanames import GenTextReturnOptMetaNames as ReturnOptions
# from tools.backend_helper import read_pdfs, listing_docs
from tools.frontend_helper import initialize_db_client, get_db_results_main, get_db_results_terms, get_db_results_changes, open_pdf, get_model
import streamlit.components.v1 as components

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
# st.image("assets/images/scb-promo-hdr.png")

# # BANNERS GO HERE >>>>>>>
# imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

# imageUrls = [
#     "https://images.unsplash.com/photo-1624704765325-fd4868c9702e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=764&q=80",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/loans/car-loans/promotion/images/2566/honda-car-loan/nov/honda-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/loans/home-loans/promotions/images/2566/home-builder/nov/home-builder-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/cards/debit-cards/promotions/images/2566/debit-inter-spend/nov/debit-inter-spend-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/cards/debit-cards/promotions/images/2566/lets-acquisition/oct/lets-acquisition-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/cards/debit-cards/promotions/images/2566/foodpanda/lets-foodpanda-day/oct/foodpanda-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/cards/debit-cards/promotions/images/2566/debit-king-power/oct/king-power-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/cards/debit-cards/promotions/images/2566/lets-sf-popcorn/sf-popcorn-detail2.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/loans/car-loans/promotion/images/2566/motor-expo/motor-expo-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/loans/home-loans/promotions/images/2566/scg/scg-detail.jpg",
#     "https://www.scb.co.th/content/dam/scb/personal-banking/loans/personal-loans/promotions/images/2566/scbm/speedy-cash/scbm-speedy-cash/aug/scbm-speedy-cash-detail.jpg",
#     "",

# ]
# selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)



st.header("Ask about Dhipaya's HR Policy 💬")
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

def format_text_main(i):
    # Chunk 1
    translated_documents = docs_main[i].chunk
    currentdoc = docs_main[i]
    full_text = f"""
    Section: {currentdoc.Section} \n
    Description: {currentdoc.Description} \n
    Content: \n
    {translated_documents}
    """
    return full_text

def format_text_terms(i):
    # Chunk 1
    meaning = docs_terms[i].Meaning_th
    terms = docs_terms[i].Term_th
    full_text = f"""
    Merms: {terms} \n
    Meaning: {meaning} \n
    """
    return full_text


def format_text_changes(i):
    # Chunk 1
    file_name = docs_changes[i].file_name
    Description = docs_changes[i].Description
    full_text = f"""
    Merms: {file_name} \n
    Meaning: {Description} \n
    """
    return full_text


# show user input
if user_question := st.text_input(
    "ask away:"
):
    with st.spinner('Initializing...'):
        logging.info(user_question)
        model_response_placeholder = st.empty()
        thai_spinner = st.empty()
        model_response_placeholder_thai = st.empty()
        # Placeholder for dynamic text and spinner

        # model_response_placeholder.text_area(label="Model Response", value="generating...", height=300)
        translated_user_input = translate_large_text(user_question,
                                                    translate_to_thai, False)
        model = get_model(model_name="sentence-transformers/all-MiniLM-L6-v2",
                        max_seq_length=384)
    
        docs_main = get_db_results_main(translated_user_input, model,
                            collection_name="dhipaya_main_doc",
                            n_results=4)
        docs_terms = get_db_results_terms(translated_user_input, model,
                            collection_name="dhipaya_term_names",
                            n_results=4)
        docs_changes = get_db_results_changes(translated_user_input, model,
                            collection_name="dhipaya_changes_policy",
                            n_results=4)

        
    with st.spinner('Initializing llm model and formulating prompt ...'):
        st.markdown("#### Main Documents")
        for i in range(4):
            chunk_label = f"ข้อมูลเกี่ยวข้องในเอกสารอันดับ {i + 1}"
            source_button_label = f"View source {i + 1}"

            # Display the text area for the chunk
            st.text_area(label=chunk_label, value=format_text_main(i), height=300)

            # # Create a button to view the source; if clicked, open the PDF
            # if st.button(source_button_label):
            #     open_pdf(docs[i].sources, docs[i].pagesno)

        st.markdown("#### Relevent Terms")
        for i in range(4):
            chunk_label = f"คำและความหมายที่เกี่ยวข้อง {i + 1}"
            source_button_label = f"View source {i + 1}"

            # Display the text area for the chunk
            st.text_area(label=chunk_label, value=format_text_terms(i), height=100)

        st.markdown("#### Changes in Documents")
        for i in range(4):
            chunk_label = f"เอกสารเกี่ยวข้องที่มีการเปลี่ยนแปลง {i + 1}"
            source_button_label = f"View source {i + 1}"

            # Display the text area for the chunk
            st.text_area(label=chunk_label, value=format_text_changes(i), height=300)


        print('docs_main' + "*" * 5)
        print(docs_main)
        print("*" * 5)

        print('docs_terms' + "*" * 5)
        print(docs_terms)
        print("*" * 5)

        print('docs_changes' + "*" * 5)
        print(docs_changes)
        print("*" * 5)

    
        model_params = {
            GenParams.DECODING_METHOD: 'greedy',
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 300,
            # GenParams.RANDOM_SEED: 42,
            # GenParams.TEMPERATURE: 0.7,
            GenParams.REPETITION_PENALTY: 1,
            GenParams.RETURN_OPTIONS: {ReturnOptions.GENERATED_TOKENS: True, ReturnOptions.GENERATED_TOKENS:True, ReturnOptions.INPUT_TOKENS: True}
        }
        model_llm = Model(ModelTypes.LLAMA_2_70B_CHAT.value, params=model_params, credentials=creds, project_id=project_id)

        knowledge_based_template = (
            open("assets/llama2-prompt-template-rag.txt",
                encoding="utf8").read().format(
            )
        )
        custom_prompt = PromptTemplate(template=knowledge_based_template,
                                    input_variables=["context", "terms", "changes", "question"])
        chunks_combined = ""
        for i in range(len(docs_main)):
            chunks_combined += f"chunk {i} \n" + docs_main[i].text_to_encode + "\n"

        terms_combined = ""
        for i in range(len(docs_terms)):
            terms_combined += f"Term {i} \n" + docs_terms[i].text_to_encode + "\n"

        changes_combined = ""
        for i in range(len(docs_terms)):
            changes_combined += f"Changes {i} \n" + docs_changes[i].text_to_encode + "\n"

        formated_prompt = custom_prompt.format(question=translated_user_input,
                                               context=chunks_combined,
                                               terms=terms_combined,
                                               changes=changes_combined)
        logging.info(formated_prompt)

        logging.info("start generate")

    with model_response_placeholder.container():
        st.markdown('---')
        st.markdown('#### Response English:')
        st.markdown('---')
    
    full_response = []
    # Loop through the chunks streamed back from the API call
    count = 0
    curr_len = 50
    for response in model_llm.generate_text_stream(formated_prompt):
        print("here", response)
        answer = response
        count = count + 1
        wordstream = str(answer)
        if wordstream:
            full_response.append(wordstream)
            result = "".join(full_response).strip()

            # This streaming_box is a st.empty from the display
            
            with model_response_placeholder.container():
                # if len(result) > curr_len:
                print(len(result))
                print("curlen", curr_len)
                
                st.markdown('---')
                st.markdown('#### Response English:')
                st.markdown(result)
                # st.markdown(translate_large_text(result,translate_to_thai, True))
                st.markdown('---')
                curr_len += 150
    print("Full", "".join(full_response).strip())
    logging.info("end generate")
    with thai_spinner:
        with st.spinner('Generating to Thai ...'):
            final_streamed_result = translate_large_text("".join(full_response).strip(),translate_to_thai, True, max_length=250)
            print(final_streamed_result)
            with model_response_placeholder_thai.container():
                st.markdown('---')
                st.markdown('#### Response Thai:')
                st.markdown(final_streamed_result + ".✔️")
                st.markdown('---')

    st.write()

