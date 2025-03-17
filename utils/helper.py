import string
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

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
# English alphabet
alphabet_lower = "abcdefghijklmnopqrstuvwxyz"
alphabet_upper = alphabet_lower.upper()

# Official Vietnamese consonants: b, c, d, đ, g, h, k, l, m, n, p, q, r, s, t, v, x
consonants_lower = "bcdđghklmnpqrstvx"
consonants_upper = consonants_lower.upper()

# Punctuation
allowed_punctuations = string.punctuation + " "

# Digits
digits = "0123456789"

# Combine all allowed characters
allowed_vietnamese = "".join(
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


def is_valid_vietnamese(sentence: str) -> bool:
    """
    Check if a Vietnamese sentence contains only allowed characters.

    Args:
        sentence (str): The Vietnamese sentence to validate.

    Returns:
        bool: True if all characters in the sentence are allowed, False otherwise.
    """
    return all(char in allowed_vietnamese for char in sentence)


def normalize_sentence(sentence: str) -> str:
    """
    Normalize a sentence by lowercasing and standardizing whitespace.

    Args:
        sentence (str): The sentence to normalize.

    Returns:
        str: The normalized sentence.
    """
    # Lowercase the sentence and remove extra space at the left and right of the sentence
    sentence = sentence.lower().strip()

    # Remove extra whitespace within the sentence
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


def tokenize_and_join(sentence: str) -> str:
    """
    Tokenize a sentence using NLTK and join tokens to re-struct a sentence.

    Args:
        sentence (str): The sentence to tokenize.

    Returns:
        str: A string with tokens joined by a space.
    """
    tokens = word_tokenize(sentence)
    return " ".join(tokens)
