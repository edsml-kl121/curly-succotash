# from googletrans import Translator
import json
import requests
import textwrap
from dotenv import load_dotenv
import os
from transformers import AutoProcessor, SeamlessM4TModel
import time
import joblib


load_dotenv()
neural_seek_url = os.getenv("NEURAL_SEEK_URL", None)
neural_seek_api_key = os.getenv("NEURAL_SEEK_API_KEY", None)


def translate_to_thai(sentence, choice):
    url = neural_seek_url  # Replace with your actual URL
    headers = {
        "accept": "application/json",
        "apikey": neural_seek_api_key,  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    if choice == True:
        target = "th"
    else:
        target = "en"
    data = {
        "text": [
            sentence
        ],
        "target": target
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
        output = json.loads(response.text)['translations'][0]
    except:
        print(data)
        if target == "th":
            output = translate_to_thai_facebook(sentence, True)
        elif target == "en":
            output = translate_to_thai_facebook(sentence, False)
        print("facebook translator: ", output)
        # print(response)
    return output




def cache_model_components(processor_path='cache/processor.joblib', model_path='cache/model.joblib'):
    """
    This function checks for the cached processor and model. If they are not found, it loads them from the pretrained source and caches them.
    Parameters:
    - processor_path: The path to save or load the processor cache.
    - model_path: The path to save or load the model cache.
    
    Returns:
    - processor: The loaded or cached processor.
    - model: The loaded or cached model.
    - initialization_time: The time taken to initialize or load the components.
    """
    start = time.time()

    # Load processor from cache if available, otherwise from pretrained
    if os.path.exists(processor_path):
        processor = joblib.load(processor_path)
    else:
        processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
        joblib.dump(processor, processor_path)

    # Load model from cache if available, otherwise from pretrained
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")
        joblib.dump(model, model_path)

    end = time.time()
    initialization_time = end - start

    return processor, model, initialization_time




def translate_to_thai_facebook(text, choice=True):
    # Example usage:
    processor, model, init_time = cache_model_components(processor_path='cache/processor.joblib', model_path='cache/model.joblib')
    print("Initializing time: ", init_time)
    start = time.time()
    # print("input text: ", text)
    if choice == True:
        text_inputs = processor(text=text, src_lang="eng", return_tensors="pt")
        output_tokens = model.generate(**text_inputs, tgt_lang="tha", generate_speech=False)
    else:
        text_inputs = processor(text=text, src_lang="tha", return_tensors="pt")
        output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False) 
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    # print("translated_text: ", translated_text_from_text)
    print("curr text: ", translated_text_from_text)
    end = time.time()
    print("time: ", end-start)
    return translated_text_from_text

def translate_large_text(text, translate_function, choice, max_length=500):
    """
    Break down large text, translate each part, and merge the results.

    :param text: str, The large body of text to translate.
    :param translate_function: function, The translation function to use.
    :param max_length: int, The maximum character length each split of text should have.
    :return: str, The translated text.
    """

    # Split the text into parts of maximum allowed character length.
    text_parts = textwrap.wrap(text, max_length, break_long_words=True,
                               replace_whitespace=False)

    translated_text_parts = []

    for part in text_parts:
        # Translate each part of the text.
        translated_part = translate_function(part,
                                             choice)  # Assuming 'False' is a necessary argument in the actual function.
        translated_text_parts.append(translated_part)

    # Combine the translated parts.
    full_translated_text = ' '.join(translated_text_parts)

    return full_translated_text
