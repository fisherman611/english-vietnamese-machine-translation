import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration  # type: ignore
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv
import wandb
import json
from utils.helper import TextPreprocessor
from utils.trainer import train_model

load_dotenv()


class MBart50Finetuner:
    """Class to handle fine-tuning of mBART50 model for translation tasks."""

    def __init__(self, config_path="config.json"):
        """Initialize with configuration file."""
        with open(config_path, "r") as json_file:
            cfg = json.load(json_file)

        self.args = cfg["mbart50"]["args"]
        self.lora_config = cfg["mbart50"]["lora_config"]

        # Constants
        self.max_len = self.args["max_len"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id = self.args["id"]
        self.initial_learning_rate = self.args["initial_learning_rate"]
        self.model_name = self.args["model_name"]
        self.src_lang = self.args["src_lang"]
        self.tgt_lang = self.args["tgt_lang"]
        self.wandb_project = self.args["wandb_project"]
        self.output_dir = self.args["output_dir"]
        self.name = "mbart50"

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        wandb.login(key=os.environ.get("WANDB_API"), relogin=True)
        wandb.init(project=self.wandb_project, name="mbart50-finetune-lora")

    def load_model_and_tokenizer(self):
        """Load the mBART model and tokenizer."""
        self.tokenizer = MBart50Tokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

    def load_datasets(self):
        """Load training, validation, and test datasets."""
        data_files = {
            "train": "data/train_cleaned_dataset.csv",
            "test": "data/test_cleaned_dataset.csv",
            "val": "data/val_cleaned_dataset.csv",
        }

        if self.id is not None:
            training_parts = [
                f"[{(i * 200000) + 1 if i > 0 else ''}:{(i + 1) * 200000 if i < 10 else ''}]"
                for i in range(11)
            ]
            self.train_dataset = load_dataset(
                "csv", data_files=data_files, split=f"train{training_parts[self.id]}"
            )
            self.test_dataset = load_dataset("csv", data_files=data_files, split="test")
            self.val_dataset = load_dataset(
                "csv", data_files=data_files, split="val[:20000]"
            )
        else:
            self.train_dataset = load_dataset(
                "csv", data_files=data_files, split="train[:1000000]"
            )
            self.test_dataset = load_dataset("csv", data_files=data_files, split="test[:100000]")
            self.val_dataset = load_dataset("csv", data_files=data_files, split="val[:100000]")

    def configure_lora(self):
        """Apply LoRA configuration to the model."""
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
        )
        self.model = get_peft_model(self.model, lora_config)  # type: ignore

    def finetune(self):
        """Orchestrate the fine-tuning process."""
        self.setup_wandb()
        self.load_model_and_tokenizer()
        self.load_datasets()

        preprocessor = TextPreprocessor(self.tokenizer, self.max_len, name="mbart50")
        tokenized_train_dataset = preprocessor.preprocess_dataset(self.train_dataset)
        tokenized_eval_dataset = preprocessor.preprocess_dataset(self.val_dataset)

        self.configure_lora()
        self.model.print_trainable_parameters()  # type: ignore

        train_model(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            output_dir=self.output_dir,
            initial_learning_rate=self.initial_learning_rate,
            name=self.name,
            val_dataset=self.val_dataset,
        )


if __name__ == "__main__":
    finetuner = MBart50Finetuner()
    finetuner.finetune()
