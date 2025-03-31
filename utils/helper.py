import re
import pandas as pd
from bs4 import BeautifulSoup  # For HTML cleaning
import spacy
import string
from unidecode import unidecode

# Load spaCy's English model.
nlp = spacy.load("en_core_web_sm")

# Precompile regex patterns once for efficiency.
MULTI_SPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

vowels_lower = (
    "aáàảãạ"
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
    "yýỳỷỹỵ"
)

vowels_upper = (
    "AÁÀẢÃẠ"
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
    "YÝỲỶỸỴ"
)

alphabet_lower = "abcdefghijklmnopqrstuvwxyz"
alphabet_upper = alphabet_lower.upper()

consonants_lower = "bcdđghklmnpqrstvx"
consonants_upper = consonants_lower.upper()

allowed_punctuations = string.punctuation + " "
digits = "0123456789"

# Combine all allowed characters into one string
allowed_pattern = "".join(
    sorted(
        set(
            vowels_lower
            + vowels_upper
            + alphabet_lower
            + alphabet_upper
            + consonants_lower
            + consonants_upper
            + allowed_punctuations
            + digits
        )
    )
)

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


def general_processing(sentence: str, max_length=50, filtering=True) -> str:
    """
    Clean and preprocess a sentence by removing extra spaces, HTML, and URLs.
    Filtering is applied if filtering is True.
    Returns None if the sentence exceeds max_length.
    """
    if filtering == True:
        if len(sentence.split()) > max_length:
            return None

    sentence = MULTI_SPACE_PATTERN.sub(" ", sentence).strip()
    sentence = BeautifulSoup(sentence, "html.parser").get_text(separator=" ")
    sentence = URL_PATTERN.sub("", sentence)

    return sentence


def english_sentence_processing(sentence: str, max_length=50, filtering=True) -> str:
    """
    Process an English sentence by converting non-ASCII characters to ASCII and applying general cleaning.
    """

    sentence = fix_non_ascii_characters(sentence)
    sentence = general_processing(sentence, max_length=max_length, filtering=filtering)
    return sentence


def vietnamese_sentence_processing(sentence: str, max_length=50, filtering=True) -> str:
    """
    Process a Vietnamese sentence if it contains only allowed characters and applying general cleaning.
    """

    if validate_vietnamese_sentence(sentence):
        sentence = general_processing(
            sentence, max_length=max_length, filtering=filtering
        )
        return sentence
    return None
