# from googletrans import Translator
import json
import requests
import textwrap
from dotenv import load_dotenv
import os

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
        return sentence
        # print(response)
    return output


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
