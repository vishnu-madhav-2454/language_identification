# ==============================================================
# 05_train_romanized.py
#
# STEP 5 — Fine-tune XLM-RoBERTa for Romanized Script LID.
#
# What the paper did (Section IV-E + Table 2):
#   • Same architecture as native model
#   • Different training data (Aksharantar + Bhasha-Abhijnaanam)
#   • Task: classify romanized words into 13 classes (0-12)
#   • Best result : 60.90% accuracy on romanized test set
#   • XLM-RoBERTa >> SVM (21.82%) on romanized text
#
# Run:  python 05_train_romanized.py
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
    ROMANIZED_TRAIN_CSV, ROMANIZED_TEST_CSV,
    ROMANIZED_MODEL_DIR,
    ROMANIZED_TRAINING_ARGS,
    LANGUAGE_LABELS,
    MAX_TOKEN_LENGTH,
    LOGS_DIR,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
NUM_LABELS = len(LANGUAGE_LABELS)   # 13


# ─────────────────────────────────────────────────────────────────
# DATASET  (identical structure to native; reused here)
# ─────────────────────────────────────────────────────────────────

class LanguageDataset(Dataset):
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
            padding=False,
            return_tensors=None,
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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


# ─────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────

def train():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Training: RoBERTa ROMANIZED   ║")
    print("╚══════════════════════════════════════════════════╝\n")

    os.makedirs(ROMANIZED_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,            exist_ok=True)

    # Check preprocessed files exist
    if not os.path.exists(ROMANIZED_TRAIN_CSV):
        raise FileNotFoundError(
            f"Romanized train CSV not found: {ROMANIZED_TRAIN_CSV}\n"
            "Run 03_preprocess_romanized.py first."
        )

    # ── 1. Tokenizer ──
    print(f"[1/5] Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # ── 2. Data ──
    print("[2/5] Loading romanized datasets …")
    train_df = pd.read_csv(ROMANIZED_TRAIN_CSV).dropna()
    test_df  = pd.read_csv(ROMANIZED_TEST_CSV).dropna()

    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test  rows : {len(test_df):,}")

    train_dataset = LanguageDataset(train_df, tokenizer, MAX_TOKEN_LENGTH)
    test_dataset  = LanguageDataset(test_df,  tokenizer, MAX_TOKEN_LENGTH)

    # ── 3. Model ──
    print(f"[3/5] Loading model: {BASE_MODEL_NAME} ({NUM_LABELS} classes)")
    id2label = {k: v for k, v in LANGUAGE_LABELS.items()}
    label2id = {v: k for k, v in LANGUAGE_LABELS.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    # ── 4. Training args ──
    print("[4/5] Setting up training arguments …")
    args = ROMANIZED_TRAINING_ARGS.copy()
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
        logging_steps=200,
        evaluation_strategy=args["evaluation_strategy"],
        save_strategy=args["save_strategy"],
        load_best_model_at_end=args["load_best_model_at_end"],
        metric_for_best_model=args["metric_for_best_model"],
        fp16=args["fp16"],
        dataloader_num_workers=args["dataloader_num_workers"],
        report_to="none",
    )

    # ── 5. Train ──
    print("[5/5] Starting training …\n")
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

    # ── Save ──
    print(f"\nSaving model to: {ROMANIZED_MODEL_DIR}")
    trainer.save_model(ROMANIZED_MODEL_DIR)
    tokenizer.save_pretrained(ROMANIZED_MODEL_DIR)

    # ── Evaluate ──
    print("\nRunning final evaluation on romanized test set …")
    results = trainer.evaluate(test_dataset)
    print("\n── Final Results (ROMANIZED) ──")
    for k, v in results.items():
        print(f"  {k:40s} : {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n✅ Romanized model saved to: {ROMANIZED_MODEL_DIR}")
    print("Next step: Run  python 06_transliteration.py  (or  python 07_pipeline.py)")


if __name__ == "__main__":
    train()
