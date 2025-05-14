"""
Fine-tune mBART50 for translation task - Main Module
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
import wandb
from utils.helper import TextPreprocessor
from utils.trainer import *

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

load_dotenv()

ARGUMENTS = cfg["mbart50"]["args"]
LORA_CONFIG = cfg["mbart50"]["lora_config"]

# Constants
MAX_LEN = ARGUMENTS["max_len"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID = ARGUMENTS["id"]
INITIAL_LEARNING_RATE = ARGUMENTS["initial_learning_rate"]
MODEL_NAME = ARGUMENTS["model_name"]
SRC_LANG = ARGUMENTS["src_lang"]
TGT_LANG = ARGUMENTS["tgt_lang"]
WANDB_PROJECT = ARGUMENTS["wandb_project"]
OUTPUT_DIR = ARGUMENTS["output_dir"]
NAME = 'mbart50'


def setup_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    wandb.login(key=os.environ.get("WANDB_API"), relogin=True)
    wandb.init(project=WANDB_PROJECT, name="mbart50-finetune-lora")


def load_model_and_tokenizer():
    """Load the mBART model and tokenizer."""
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    return model, tokenizer


def load_datasets():
    """Load training, validation, and test datasets."""
    data_files = {
        "train": "data/train_cleaned_dataset.csv",
        "test": "data/test_cleaned_dataset.csv",
        "val": "data/val_cleaned_dataset.csv",
    }

    if ID is not None:
        training_parts = [
            f"[{(i * 200000) + 1 if i > 0 else ''}:{(i + 1) * 200000 if i < 10 else ''}]"
            for i in range(11)
        ]

        train_dataset = load_dataset(
            "csv", data_files=data_files, split=f"train{training_parts[ID]}"
        )
        test_dataset = load_dataset("csv", data_files=data_files, split="test")
        val_dataset = load_dataset("csv", data_files=data_files, split="val[:20000]")
    else:
        train_dataset = load_dataset("csv", data_files=data_files, split="train")
        test_dataset = load_dataset("csv", data_files=data_files, split="test")
        val_dataset = load_dataset("csv", data_files=data_files, split="val")
    return train_dataset, val_dataset, test_dataset


def configure_lora(model):
    """Apply LoRA configuration to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
    )
    model = get_peft_model(model, lora_config)
    return model


def main():
    """Main function to orchestrate the fine-tuning process."""
    # Setup Weights & Biases
    setup_wandb()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()

    # Preprocess datasets
    preprocessor = TextPreprocessor(tokenizer, MAX_LEN, name="mbart50")
    tokenized_train_dataset = preprocessor.preprocess_dataset(train_dataset)
    tokenized_eval_dataset = preprocessor.preprocess_dataset(val_dataset)

    # Apply LoRA
    model = configure_lora(model)
    model.print_trainable_parameters()

    # Train the model
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        output_dir=OUTPUT_DIR,
        initial_learning_rate=INITIAL_LEARNING_RATE,
        name=NAME,
        val_dataset=val_dataset
    )


if __name__ == "__main__":
    main()
