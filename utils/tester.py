import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Union, Any
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import (
    MBart50Tokenizer,                       #type: ignore
    MBartForConditionalGeneration,          #type: ignore
    MT5ForConditionalGeneration,            #type: ignore
    MT5TokenizerFast,                       #type: ignore
)
from peft import PeftModel, PeftConfig
import evaluate

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.rule_based_mt import TransferBasedMT
from models.statistical_mt import SMTExtended, LanguageModel

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
with open("config.json", "r") as json_file:
    CONFIG = json.load(json_file)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate English-Vietnamese Machine Translation Models")
    parser.add_argument("--test_file", type=str, default='data/test_cleaned_dataset.csv', help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
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
        """Load fine-tuned MBart50 model and tokenizer."""
        try:
            model_config = CONFIG["mbart50"]["paths"]
            model = MBartForConditionalGeneration.from_pretrained(model_config["base_model_name"])
            peft_config = PeftConfig.from_pretrained(model_config["checkpoint_path"])
            model = PeftModel.from_pretrained(model, model_config["checkpoint_path"])
            tokenizer = MBart50Tokenizer.from_pretrained(model_config["checkpoint_path"])
            model.eval()
            print("Fine-tuned MBart50 loaded successfully!")
            return model.to(DEVICE), tokenizer              #type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to load fine-tuned MBart50 model: {str(e)}")

    @staticmethod
    def load_original_mbart50() -> Tuple[MBartForConditionalGeneration, MBart50Tokenizer]:
        """Load original MBart50 model and tokenizer."""
        try:
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = MBart50Tokenizer.from_pretrained(model_name)
            model.eval()
            print("Original MBart50 loaded successfully!")
            return model.to(DEVICE), tokenizer              #type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to load original MBart50 model: {str(e)}")

    @staticmethod
    def load_mt5() -> Tuple[MT5ForConditionalGeneration, MT5TokenizerFast]:
        """Load fine-tuned MT5 model and tokenizer."""
        try:
            model_config = CONFIG["mt5"]["paths"]
            model = MT5ForConditionalGeneration.from_pretrained(model_config["base_model_name"])
            peft_config = PeftConfig.from_pretrained(model_config["checkpoint_path"])
            model = PeftModel.from_pretrained(model, model_config["checkpoint_path"])
            tokenizer = MT5TokenizerFast.from_pretrained(model_config["checkpoint_path"])
            model.eval()
            print("Fine-tuned MT5 loaded successfully!")
            return model.to(DEVICE), tokenizer              #type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to load fine-tuned MT5 model: {str(e)}")

    @staticmethod
    def load_original_mt5() -> Tuple[MT5ForConditionalGeneration, MT5TokenizerFast]:
        """Load original MT5 model and tokenizer."""
        try:
            model_name = "google/mt5-base"
            model = MT5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = MT5TokenizerFast.from_pretrained(model_name)
            model.eval()
            print("Original MT5 loaded successfully!")
            return model.to(DEVICE), tokenizer              #type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to load original MT5 model: {str(e)}")


class Translator:
    """Handles translation using different models."""

    @staticmethod
    def translate_rbmt(text: str) -> str:
        """Translate using Rule-Based Machine Translation."""
        try:
            translator = TransferBasedMT()
            return translator.translate(text)
        except Exception as e:
            raise RuntimeError(f"RBMT translation failed: {str(e)}")

    @staticmethod
    def translate_smt(text: str, smt) -> str:
        """Translate using Statistical Machine Translation."""
        try: 
            # return smt.translate_sentence(text)
            translation = smt.infer(text)
            return translation
        except Exception as e:
            raise RuntimeError(f"SMT translation failed: {str(e)}")

    @staticmethod
    def translate_mbart50(
        model: MBartForConditionalGeneration, tokenizer: MBart50Tokenizer, text: str
    ) -> str:
        """Translate using MBart50 model (fine-tuned or original)."""
        try:
            model_config = CONFIG["mbart50"]["args"]
            tokenizer.src_lang = model_config["src_lang"]
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

            forced_bos_token_id = tokenizer.lang_code_to_id[model_config["tgt_lang"]]
            translated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=5,
            )
            return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"MBart50 translation failed: {str(e)}")

    @staticmethod
    def translate_mt5(
        model: MT5ForConditionalGeneration, tokenizer: MT5TokenizerFast, text: str
    ) -> str:
        """Translate using MT5 model (fine-tuned or original)."""
        try:
            prefix = CONFIG["mt5"]["args"]["prefix"]
            text = prefix + text
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

            translated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=5,
            )
            return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"MT5 translation failed: {str(e)}")


class Evaluator:
    """Handles evaluation of translation models."""

    @staticmethod
    def load_test_data(test_file: str) -> List[Dict[str, str]]:
        """Load test data from CSV file."""
        try:
            df = pd.read_csv(test_file)
            return [{"source": row["en"], "reference": row["vi"]} for _, row in df.iterrows()]
        except Exception as e:
            raise RuntimeError(f"Failed to load test data: {str(e)}")

    @staticmethod
    def compute_metrics(hypotheses: List[str], references: List[str], sources: List[str]) -> Dict[str, float]:
        """Compute translation evaluation metrics."""
        try:
            metrics = {}
            bleu_metric = evaluate.load("sacrebleu")
            meteor_metric = evaluate.load("meteor")
            rouge_metric = evaluate.load("rouge")
            comet_metric = evaluate.load("comet")
            bertscore_metric = evaluate.load("bertscore")
            
            # BLEU
            metrics["SacreBLEU"] = bleu_metric.compute(predictions=hypotheses, references=references)["score"] / 100        #type: ignore 

            # METEOR
            metrics["METEOR"] = meteor_metric.compute(predictions=hypotheses, references=references)["meteor"]            #type: ignore

            # ROUGE
            rouge_results = rouge_metric.compute(
                predictions=hypotheses, references=references, rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            metrics["ROUGE-1"] = rouge_results["rouge1"]          #type: ignore
            metrics["ROUGE-2"] = rouge_results["rouge2"]          #type: ignore
            metrics["ROUGE-L"] = rouge_results["rougeL"]          #type: ignore

            # BERTScore
            bertscore_results = bertscore_metric.compute(
                predictions=hypotheses, references=references, model_type="bert-base-multilingual-cased", lang="vi"
            )
            metrics["BERTScore"] = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])            #type: ignore

            # COMET
            comet_results = comet_metric.compute(predictions=hypotheses, references=references, sources=sources)
            metrics["COMET"] = sum(comet_results["scores"]) / len(comet_results["scores"])                #type: ignore

            return metrics
        except Exception as e:
            raise RuntimeError(f"Failed to compute metrics: {str(e)}")

    @staticmethod
    def evaluate_model(
        model_type: str, test_data: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Evaluate a translation model on test data."""
        hypotheses, references, sources = [], [], []

        try:
            if model_type == "rbmt":
                for item in tqdm(test_data, desc="Translating with RBMT"):
                    translation = Translator.translate_rbmt(item["source"])
                    hypotheses.append(translation)
                    references.append(item["reference"])
                    sources.append(item["source"])

            elif model_type == "smt":
                for item in tqdm(test_data, desc="Translating with SMT"):
                    smt = ModelLoader.load_smt()
                    translation = Translator.translate_smt(item["source"], smt)
                    hypotheses.append(translation)
                    references.append(item["reference"])
                    sources.append(item["source"])

            elif model_type == "mbart50":
                model, tokenizer = ModelLoader.load_mbart50()
                for item in tqdm(test_data, desc="Translating with fine-tuned mBART50"):
                    translation = Translator.translate_mbart50(model, tokenizer, item["source"])
                    hypotheses.append(translation)
                    references.append(item["reference"])
                    sources.append(item["source"])

            elif model_type == "original_mbart50":
                model, tokenizer = ModelLoader.load_original_mbart50()
                for item in tqdm(test_data, desc="Translating with original mBART50"):
                    translation = Translator.translate_mbart50(model, tokenizer, item["source"])
                    hypotheses.append(translation)
                    references.append(item["reference"])
                    sources.append(item["source"])

            elif model_type == "mt5":
                model, tokenizer = ModelLoader.load_mt5()
                for item in tqdm(test_data, desc="Translating with fine-tuned MT5"):
                    translation = Translator.translate_mt5(model, tokenizer, item["source"])
                    hypotheses.append(translation)
                    references.append(item["reference"])
                    sources.append(item["source"])

            elif model_type == "original_mt5":
                model, tokenizer = ModelLoader.load_original_mt5()
                for item in tqdm(test_data, desc="Translating with original MT5"):
                    translation = Translator.translate_mt5(model, tokenizer, item["source"])
                    hypotheses.append(translation)
                    references.append(item["reference"])
                    sources.append(item["source"])

            return hypotheses, references, Evaluator.compute_metrics(hypotheses, references, sources) if hypotheses else {}
        except Exception as e:
            raise RuntimeError(f"Evaluation failed for {model_type}: {str(e)}")


def main():
    """Main function to run model evaluation."""
    args = parse_arguments()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        test_data = Evaluator.load_test_data(args.test_file)
        model_types = ["rbmt", "smt" "mbart50", "original_mbart50", "mt5", "original_mt5"]
        all_results = {}

        for model_type in model_types:
            print(f"\nEvaluating {model_type}...")
            hypotheses, references, metrics = Evaluator.evaluate_model(model_type, test_data)

            if metrics:
                all_results[model_type] = metrics
                print(f"Metrics for {model_type}:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")

                # Save translations
                translations = [
                    {"source": item["source"], "reference": ref, "hypothesis": hyp}
                    for item, ref, hyp in zip(test_data, references, hypotheses)
                ]
                with open(
                    Path(args.output_dir) / f"{model_type}_translations.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(translations, f, ensure_ascii=False, indent=2)

        # Save all metrics
        with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()