import os
import sys
import json
import argparse
from typing import Tuple, Union, Dict, Any
from pathlib import Path

import torch
from transformers import (
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5TokenizerFast,
)
from peft import PeftModel, PeftConfig

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.rule_based_mt import TransferBasedMT
from models.statistical_mt import SMTExtended, LanguageModel

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration once
with open("config.json", "r") as json_file:
    CONFIG = json.load(json_file)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="English-Vietnamese Machine Translation Inference")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["rbmt", "smt", "mbart50", "mt5"],
        required=True,
        help="Type of model to use for translation",
    )
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    return parser.parse_args()


class ModelLoader:
    """Handles loading of translation models."""

    @staticmethod
    def load_smt() -> None:
        """Load Statistical Machine Translation model."""
        try:
            smt = SMTExtended()
            model_dir = "checkpoints"
            if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "phrase_table.pkl")):
                print("Loading existing model...")
                smt.load_model()
            else:
                print("Training new smt...")
                stats = smt.train()
                print(f"Training complete: {stats}")
            print("SMT model loaded successfully!")
            return smt
        except Exception as e:
            raise RuntimeError(f"Failed to load SMT model: {str(e)}")

    @staticmethod
    def load_mbart50() -> Tuple[MBartForConditionalGeneration, MBart50Tokenizer]:
        """Load MBart50 model and tokenizer."""
        try:
            model_config = CONFIG["mbart50"]["paths"]
            model = MBartForConditionalGeneration.from_pretrained(model_config["base_model_name"])
            model = PeftModel.from_pretrained(model, model_config["checkpoint_path"])
            tokenizer = MBart50Tokenizer.from_pretrained(model_config["checkpoint_path"])
            model.eval()
            print("MBart50 loaded successfully!")
            return model.to(DEVICE), tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load MBart50 model: {str(e)}")

    @staticmethod
    def load_mt5() -> Tuple[MT5ForConditionalGeneration, MT5TokenizerFast]:
        """Load MT5 model and tokenizer."""
        try:
            model_config = CONFIG["mt5"]["paths"]
            model = MT5ForConditionalGeneration.from_pretrained(model_config["base_model_name"])
            model = PeftModel.from_pretrained(model, model_config["checkpoint_path"])
            tokenizer = MT5TokenizerFast.from_pretrained(model_config["checkpoint_path"])
            model.eval()
            print("MT5 loaded successfully!")
            return model.to(DEVICE), tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load MT5 model: {str(e)}")


class Translator:
    """Handles translation using different models."""

    @staticmethod
    def translate_rbmt(text: str) -> str:
        """Translate using Rule-Based Machine Translation."""
        try:
            return TransferBasedMT().translate(text)
        except Exception as e:
            raise RuntimeError(f"RBMT translation failed: {str(e)}")

    @staticmethod
    def translate_smt(text: str, smt) -> str:
        """Translate using Statistical Machine Translation."""
        try: 
            return smt.translate_sentence(text)
            translation = smt.infer(text)
            return translation
        except Exception as e:
            raise RuntimeError(f"SMT translation failed: {str(e)}")

    @staticmethod
    def translate_mbart50(
        text: str, model: MBartForConditionalGeneration, tokenizer: MBart50Tokenizer
    ) -> str:
        """Translate using MBart50 model with batch processing."""
        try:
            model_config = CONFIG["mbart50"]["args"]
            tokenizer.src_lang = model_config["src_lang"]
            inputs = tokenizer([text], return_tensors="pt", padding=True)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

            with torch.no_grad():  # Disable gradient computation for inference
                translated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    forced_bos_token_id=tokenizer.lang_code_to_id[model_config["tgt_lang"]],
                    max_length=128,
                    num_beams=5,
                )
            return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        except Exception as e:
            raise RuntimeError(f"MBart50 translation failed: {str(e)}")

    @staticmethod
    def translate_mt5(
        text: str, model: MT5ForConditionalGeneration, tokenizer: MT5TokenizerFast
    ) -> str:
        """Translate using MT5 model with batch processing."""
        try:
            prefix = CONFIG["mt5"]["args"]["prefix"]
            inputs = tokenizer([prefix + text], return_tensors="pt", padding=True)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

            with torch.no_grad():  # Disable gradient computation for inference
                translated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=128,
                    num_beams=5,
                )
            return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        except Exception as e:
            raise RuntimeError(f"MT5 translation failed: {str(e)}")


def main():
    """Main function to run translation."""
    args = parse_arguments()

    try:
        if args.model_type == "rbmt":
            translation = Translator.translate_rbmt(args.text)
        elif args.model_type == "smt":
            smt = ModelLoader.load_smt()
            translation = Translator.translate_smt(args.text, smt)
        elif args.model_type == "mbart50":
            model, tokenizer = ModelLoader.load_mbart50()
            translation = Translator.translate_mbart50(args.text, model, tokenizer)
        else:  # mt5
            model, tokenizer = ModelLoader.load_mt5()
            translation = Translator.translate_mt5(args.text, model, tokenizer)
        
        print(f"Translation: {translation}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
