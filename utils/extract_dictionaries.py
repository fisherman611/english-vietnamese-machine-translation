import pandas as pd 
import json 
import nltk 
from nltk.tokenize import word_tokenize
import spacy 
import contractions
import string 
import re
from collections import defaultdict
from tqdm.auto import tqdm
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='en', target='vi')

def translate_word(word: str) -> str: 
    try: 
        translation = translator.translate(word).lower()
        return translation 
    except Exception as e: 
        return "N/A"

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('data/train_cleaned_dataset.csv')

def preprocessing(sentence: str) -> list[str]:
    """
    Preprocess the input sentence: remove named entities, lowercase, expand contractions, 
    and tokenize into a list of words.
    
    Args:
        sentence (str): The input sentence to preprocess.
    
    Returns:
        list[str]: A list of preprocessed tokens.
    """
    if not sentence or not sentence.strip():
        return []

    # Step 1: Remove named entities
    doc = nlp(sentence)
    entities = sorted(doc.ents, key=lambda ent: len(ent.text), reverse=True)  # Sort by length to handle nested entities
    for ent in entities:
        sentence = sentence.replace(ent.text, "")  # Remove the entity entirely

    # Step 2: Clean up extra spaces after entity removal
    sentence = " ".join(sentence.split()).strip()

    # Step 3: Remove all numbers using regex
    sentence = re.sub(r'\d+', '', sentence)

    # Step 4: Lowercase the sentence
    sentence = sentence.lower()

    # Step 5: Expand contractions (e.g., "don't" -> "do not")
    sentence = contractions.fix(sentence)

    # Step 6: Remove punctuation 
    translator = str.maketrans({p: ' ' for p in string.punctuation})
    sentence = sentence.translate(translator)
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    # Step 7: Tokenize into words
    words = word_tokenize(sentence)

    return words

english_sentences = df['en'].to_list()


# Extract English vocabs: 

eng_vocabs = defaultdict(int)
for eng_sent in tqdm(english_sentences): 
    words = preprocessing(eng_sent)
    for word in words:
        eng_vocabs[word] += 1

dictionary = {}
for word in list(eng_vocabs.keys()): 
    dictionary[word] = translate_word(word)

dictionary = {k: v for k, v in dictionary.items() if v != "N/A"}
sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[0])
final_dict = dict(sorted_dictionary)

output_file = "data/en_vi_dictionary.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_dict, f, ensure_ascii=False, indent=4)

print(f"English vocabulary saved to {output_file}")


    
