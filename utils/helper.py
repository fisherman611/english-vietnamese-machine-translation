import re
import pandas as pd
from bs4 import BeautifulSoup  # For HTML cleaning
import spacy
import string
from unidecode import unidecode
from typing import List, Tuple, Any
from tqdm.auto import tqdm

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

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


def english_sentence_processing(sentence: str,
                                max_length=50,
                                filtering=True) -> str:
    """
    Process an English sentence by converting non-ASCII characters to ASCII and applying general cleaning.
    """

    sentence = fix_non_ascii_characters(sentence)
    sentence = general_processing(sentence,
                                  max_length=max_length,
                                  filtering=filtering)
    return sentence


def vietnamese_sentence_processing(sentence: str,
                                   max_length=50,
                                   filtering=True) -> str:
    """
    Process a Vietnamese sentence if it contains only allowed characters and applying general cleaning.
    """

    if validate_vietnamese_sentence(sentence):
        sentence = general_processing(sentence,
                                      max_length=max_length,
                                      filtering=filtering)
        return sentence
    return None


class TextPreprocessor:

    def __init__(self, tokenizer, max_length, name):
        """
        Initializes the text preprocessor with a tokenizer and maximum sequence length.

        Args:
            tokenizer: The tokenizer used for tokenizing input and target text.
            max_length: The maximum length for tokenized sequences (inputs and targets).
            name: "mt5" or "mbart50"
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.name = name

    def preprocess_function(self, examples):
        """
        Tokenizes and formats a batch of examples for sequence-to-sequence training.

        Args:
            examples: A dictionary with "en" (English source text) and "vi" (Vietnamese target text) keys.

        Returns:
            A dictionary containing tokenized inputs and labels with necessary padding/truncation.
        """
        # Get source and target text
        if self.name == "mbart50":
            inputs = examples["en"]
        else:
            PREFIX = cfg[self.name]["args"]["prefix"]
            inputs = [PREFIX + example for example in examples["en"]]
        targets = examples["vi"]

        # Tokenize both inputs and targets with padding and truncation
        model_inputs = self.tokenizer(
            inputs,
            text_target=targets,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Replace padding token ids in labels with -100 to ignore them in loss computation
        model_inputs["labels"] = [[
            (t if t != self.tokenizer.pad_token_id else -100) for t in seq
        ] for seq in model_inputs["labels"]]

        # Preserve original texts for reference or debugging
        if self.name == "mbart50":
            model_inputs["en"] = examples["en"]
        else:
            PREFIX = cfg[self.name]["args"]["prefix"]
            model_inputs["en"] = [
                PREFIX + example for example in examples["en"]
            ]
        model_inputs["vi"] = examples["vi"]

        return model_inputs

    def preprocess_dataset(self, dataset):
        """
        Applies preprocessing to the entire dataset using the `preprocess_function`.

        Args:
            dataset: A Hugging Face Dataset object containing examples with "en" and "vi" keys.

        Returns:
            A tokenized and formatted dataset ready for training.
        """
        return dataset.map(self.preprocess_function,
                           batched=True,
                           remove_columns=dataset.column_names)


def clean_sentence(sentence: str) -> str:
    """Remove punctuation and convert to lowercase."""
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    cleaned = sentence.translate(translator)
    # Convert to lowercase
    return cleaned.lower()


def build_corpus(csv_file: str) -> List[Tuple[List[str], List[str]]]:
    """Read CSV, clean sentences, and create corpus."""
    df = pd.read_csv(csv_file)
    corpus = []
    for _, row in tqdm(df.iterrows()):
        # Clean and tokenize
        eng_cleaned = clean_sentence(row['en'].strip())
        vi_cleaned = clean_sentence(row['vi'].strip())
        eng_words = eng_cleaned.split()
        vi_words = vi_cleaned.split()
        if eng_words and vi_words:  # Skip empty sentences
            corpus.append((eng_words, vi_words))

    return corpus
