import re
import pandas as pd
from bs4 import BeautifulSoup  # For HTML cleaning
import spacy
import truecase
import string
from unidecode import unidecode

# Load spaCy's English model.
nlp = spacy.load("en_core_web_sm")

# Precompile regex patterns once for efficiency.
MULTI_SPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

vowels_lower = ("aáàảãạ"
                "ăắằẳẵặ"
                "âấầẩẫậ"
                "eéèẻẽẹ"
                "êếềểễệ"
                "iíìỉĩị"
                "oóòỏõọ"
                "ôốồổỗộ"
                "ơớờởỡợ"
                "uúùủũụ"
                "ưứừửữự"
                "yýỳỷỹỵ")

vowels_upper = ("AÁÀẢÃẠ"
                "ĂẮẰẲẴẶ"
                "ÂẤẦẨẪẬ"
                "EÉÈẺẼẸ"
                "ÊẾỀỂỄỆ"
                "IÍÌỈĨỊ"
                "OÓÒỎÕỌ"
                "ÔỐỒỔỖỘ"
                "ƠỚỜỞỠỢ"
                "UÚÙỦŨỤ"
                "ƯỨỪỬỮỰ"
                "YÝỲỶỸỴ")

alphabet_lower = "abcdefghijklmnopqrstuvwxyz"
alphabet_upper = alphabet_lower.upper()

consonants_lower = "bcdđghklmnpqrstvx"
consonants_upper = consonants_lower.upper()

allowed_punctuations = string.punctuation + " "
digits = "0123456789"

# Combine all allowed characters into one string
allowed_pattern = "".join(
    sorted(
        set(vowels_lower + vowels_upper + alphabet_lower + alphabet_upper +
            consonants_lower + consonants_upper + allowed_punctuations +
            digits)))

# Escape the allowed characters so that regex meta-characters are taken literally.
escaped_allowed = re.escape(allowed_pattern)
regex_pattern = rf"^[{escaped_allowed}]+$"

# Compile the regex
VIETNAMESE_ALLOWED_PATTERN = re.compile(regex_pattern)


def validate_vietnamese_sentence(sentence: str) -> bool:
    """
    Return True if the Vietnamese sentence contains only allowed characters; otherwise, False.
    """
    return VIETNAMESE_ALLOWED_PATTERN.fullmatch(sentence) is not None


def fix_non_ascii_characters(sentence: str) -> str:
    """
    Replace non-ASCII characters in the sentence with their closest ASCII equivalents.
    """
    return unidecode(sentence)


def general_processing(sentence: str,
                       max_words=50,
                       apply_truecase=False,
                       selective_case=False) -> str:
    """
    Clean and preprocess a sentence by removing extra spaces, HTML, and URLs,
    then apply casing (truecase, selective lowercasing, or full lowercase).
    Returns None if the sentence exceeds max_words.
    """
    if len(sentence.split()) > max_words:
        return None

    sentence = MULTI_SPACE_PATTERN.sub(" ", sentence).strip()
    sentence = BeautifulSoup(sentence, "html.parser").get_text(separator=" ")
    sentence = URL_PATTERN.sub("", sentence)

    if apply_truecase:
        sentence = truecase.get_true_case(sentence)
    elif selective_case:
        doc = nlp(sentence)
        sentence = ' '.join([
            token.text if token.pos_ == "PROPN" else token.text.lower()
            for token in doc
        ])
    else:
        sentence = sentence.lower()

    return sentence

# print(general_processing(fix_non_ascii_characters('Sergio Ramós played at Laliga'), selective_case=True))