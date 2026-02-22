# ==============================================================
# 04_train_native.py
#
# STEP 4 — Fine-tune XLM-RoBERTa for Native Script LID.
#
# What the paper did (Section IV-E):
#   • Base model  : xlm-roberta-base (125M params)
#   • Task        : 13-class classification (0=English, 1-12=Indian)
#   • Loss        : Cross-Entropy
#   • Optimizer   : Adam
#   • LR          : 2e-5
#   • Epochs      : 10
#   • Train batch : 1280 (paper) → we use 16/GPU with accumulation
#   • Mixed prec. : FP16
#   • Result      : 99.54% accuracy on native test set
#
# Run:  python 04_train_native.py
# ==============================================================

import os
import sys

# Must be set BEFORE importing transformers to suppress broken TensorFlow DLL
os.environ["USE_TF"]    = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_MODEL_NAME,
    NATIVE_TRAIN_CSV, NATIVE_TEST_CSV,
    NATIVE_MODEL_DIR,
    NATIVE_TRAINING_ARGS,
    LANGUAGE_LABELS,
    MAX_TOKEN_LENGTH,
    LOGS_DIR,
)

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
NUM_LABELS = len(LANGUAGE_LABELS)   # 13 (0–12)


# ─────────────────────────────────────────────────────────────────
# CUSTOM PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────

class LanguageDataset(Dataset):
    """
    Wraps a pandas DataFrame for PyTorch.

    The DataFrame has two columns:
        Sentence  : cleaned text
        Language  : integer label (0-12)
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data.loc[idx, "Sentence"])
        label    = int(self.data.loc[idx, "Language"])

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_len,
            padding=False,          # DataCollatorWithPadding pads later
            return_tensors=None,    # return plain lists (faster)
        )

        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         label,
        }


# ─────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")


def compute_metrics(eval_pred):
    """
    Custom metric function called by HuggingFace Trainer after
    each evaluation epoch.  Returns accuracy and macro-F1.
    (Paper used accuracy as primary metric.)
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro",
    )
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_data():
    """Load preprocessed CSVs. Returns (train_df, test_df)."""
    if not os.path.exists(NATIVE_TRAIN_CSV):
        raise FileNotFoundError(
            f"Training CSV not found: {NATIVE_TRAIN_CSV}\n"
            "Run 02_preprocess_native.py first."
        )

    print(f"  Loading train: {NATIVE_TRAIN_CSV}")
    train_df = pd.read_csv(NATIVE_TRAIN_CSV).dropna()

    print(f"  Loading test : {NATIVE_TEST_CSV}")
    test_df  = pd.read_csv(NATIVE_TEST_CSV).dropna()

    print(f"  Train size : {len(train_df):,}")
    print(f"  Test size  : {len(test_df):,}")
    print(f"  Num labels : {NUM_LABELS}")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────

def train():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Training: RoBERTa NATIVE      ║")
    print("╚══════════════════════════════════════════════════╝\n")

    os.makedirs(NATIVE_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,         exist_ok=True)

    # ── 1. Load tokenizer ──
    print(f"[1/5] Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # ── 2. Load and wrap datasets ──
    print("[2/5] Loading datasets …")
    train_df, test_df = load_data()

    train_dataset = LanguageDataset(train_df, tokenizer, MAX_TOKEN_LENGTH)
    test_dataset  = LanguageDataset(test_df,  tokenizer, MAX_TOKEN_LENGTH)

    # ── 3. Load pre-trained model with classification head ──
    print(f"[3/5] Loading model: {BASE_MODEL_NAME}")
    print(f"      Classification head: {NUM_LABELS} classes")

    # id2label and label2id for HuggingFace metadata
    id2label = {k: v for k, v in LANGUAGE_LABELS.items()}
    label2id = {v: k for k, v in LANGUAGE_LABELS.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    # ── 4. Training arguments ──
    print("[4/5] Setting up training arguments …")
    args = NATIVE_TRAINING_ARGS.copy()

    # gradient_accumulation_steps is now set in config.py
    # effective_batch = per_device_train_batch_size × gradient_accumulation_steps
    # e.g. 4 × 320 = 1280 (same as paper, tuned for 4 GB VRAM)
    grad_accum = args.get("gradient_accumulation_steps", 320)

    training_args = TrainingArguments(
        output_dir=args["output_dir"],
        num_train_epochs=args["num_train_epochs"],
        learning_rate=args["learning_rate"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        per_device_eval_batch_size=args["per_device_eval_batch_size"],
        gradient_accumulation_steps=grad_accum,
        warmup_steps=args["warmup_steps"],
        weight_decay=args["weight_decay"],
        logging_dir=args["logging_dir"],
        logging_steps=500,
        evaluation_strategy=args["evaluation_strategy"],
        save_strategy=args["save_strategy"],
        load_best_model_at_end=args["load_best_model_at_end"],
        metric_for_best_model=args["metric_for_best_model"],
        fp16=args["fp16"],
        dataloader_num_workers=args["dataloader_num_workers"],
        report_to="none",   # disable wandb / tensorboard by default
    )

    # ── 5. Trainer ──
    print("[5/5] Starting training …\n")
    print(f"      Model      : {BASE_MODEL_NAME}")
    print(f"      Epochs     : {training_args.num_train_epochs}")
    print(f"      LR         : {training_args.learning_rate}")
    print(f"      Eff. batch : {training_args.per_device_train_batch_size * grad_accum}")
    print(f"      FP16       : {training_args.fp16}")
    print()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Save best model + tokenizer ──
    print(f"\nSaving model to: {NATIVE_MODEL_DIR}")
    trainer.save_model(NATIVE_MODEL_DIR)
    tokenizer.save_pretrained(NATIVE_MODEL_DIR)

    # ── Final evaluation ──
    print("\nRunning final evaluation on test set …")
    results = trainer.evaluate(test_dataset)
    print("\n── Final Results (NATIVE) ──")
    for k, v in results.items():
        print(f"  {k:40s} : {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n✅ Native model saved to: {NATIVE_MODEL_DIR}")
    print("Next step: Run  python 05_train_romanized.py")


if __name__ == "__main__":
    train()
