import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import numpy as np
import torch
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from utils.helper import TextPreprocessor

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)


class SaveBestModelCallback(TrainerCallback):

    def __init__(self, output_dir, name="mbart50"):
        """
        Callback to save the model with the best composite score, overriding previous best.

        Args:
            output_dir: Directory to save checkpoints
            name: Name for checkpoint directory
        """
        self.output_dir = output_dir
        self.name = name
        self.prefix = f"best_{self.name}"
        self.best_composite_score = float("-inf")
        self.checkpoint_dir = os.path.join(self.output_dir, self.prefix)
        self.metric_weights = cfg["metric_weights"]
        self.best_metrics = {
            metric: float("-inf")
            for metric in self.metric_weights
        }

    def calculate_composite_score(self, metrics):
        """Calculate a weighted composite score from multiple metrics."""
        composite_score = 0.0
        for metric, weight in self.metric_weights.items():
            metric_value = metrics.get(f"eval_{metric}", 0.0)
            composite_score += metric_value * weight
        return composite_score

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_composite_score = self.calculate_composite_score(metrics)
        if current_composite_score > self.best_composite_score:
            print(f"New best composite score: {current_composite_score:.4f} "
                  f"(previous: {self.best_composite_score:.4f})")
            self.best_composite_score = current_composite_score
            for metric in self.metric_weights:
                metric_value = metrics.get(f"eval_{metric}", 0.0)
                if metric_value > self.best_metrics[metric]:
                    self.best_metrics[metric] = metric_value
            # Remove existing checkpoint if it exists
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            # Save new best model to same checkpoint directory
            self.trainer.save_model(self.checkpoint_dir)
            print(f"Saved best model to: {self.checkpoint_dir}")
            print("Current metric values:")
            for metric, value in self.best_metrics.items():
                print(f"  {metric}: {value:.4f}")
        control.should_save = False

    def on_train_end(self, args, state, control, **kwargs):
        if os.path.exists(self.checkpoint_dir):
            print(
                f"Training complete. Best model saved at: {self.checkpoint_dir}"
            )
            print(f"Best composite score: {self.best_composite_score:.4f}")
            print("Best metric values:")
            for metric, value in self.best_metrics.items():
                print(f"  {metric}: {value:.4f}")


def compute_metrics(eval_preds, tokenizer, tokenized_eval_dataset,
                    val_dataset):
    """
    Compute evaluation metrics for translation.

    Args:
        eval_preds: Tuple of predictions and labels
        tokenizer: Tokenizer used for decoding
        tokenized_eval_dataset: Tokenized evaluation dataset

    Returns:
        Dictionary of metric scores
    """
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    comet_metric = evaluate.load("comet")
    bert_metric = evaluate.load("bertscore")

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_no_ignore = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_no_ignore,
                                            skip_special_tokens=True)
    source_texts = val_dataset["en"]

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],
    )
    rouge_scores = rouge_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    meteor = meteor_metric.compute(predictions=decoded_preds,
                                   references=[[label]
                                               for label in decoded_labels])
    bert_score = bert_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],
        model_type="bert-base-multilingual-cased",
        lang="vi",
    )
    comet = comet_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        sources=source_texts,
    )
    return {
        "bleu": bleu["score"] / 100,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "meteor": meteor["meteor"],
        "bertscore": sum(bert_score["f1"]) / len(bert_score["f1"]),
        "comet": sum(comet["scores"]) / len(comet["scores"]),
    }


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    initial_learning_rate,
    name,
    val_dataset,
):
    """
    Train the mBART model with the provided datasets and configurations.

    Args:
        model: The mBART model to train
        tokenizer: The tokenizer for preprocessing
        train_dataset: Tokenized training dataset
        eval_dataset: Tokenized evaluation dataset
        output_dir: Directory to save checkpoints
        initial_learning_rate: Initial learning rate for training
        model_name: Name of the using model
        val_dataset: Validation dataset
    """
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    arguments = cfg[name]['args']
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        save_strategy="no",
        save_safetensors=True,
        learning_rate=initial_learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=arguments["warmup_steps"],
        per_device_train_batch_size=arguments["per_device_train_batch_size"],
        num_train_epochs=arguments["num_train_epochs"],
        weight_decay=arguments["weight_decay"],
        fp16=True if name == 'mbart50' else False,
        bf16=True if name == 'mt5' else False,
        eval_strategy="epoch",
        per_device_eval_batch_size=arguments["per_device_eval_batch_size"],
        predict_with_generate=True,
        logging_strategy="epoch",
        logging_first_step=True,
        report_to="wandb",
        seed=42,
    )

    save_callback = SaveBestModelCallback(output_dir=output_dir, name=name)

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, eval_dataset, val_dataset),
        callbacks=[save_callback],
    )

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, SaveBestModelCallback):
            callback.trainer = trainer

    # Start training
    trainer.train()
