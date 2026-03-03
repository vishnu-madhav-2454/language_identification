# ==============================================================
# 05_train_indicbert.py
#
# STEP 5 — Fine-tune IndicBERT for Romanized Text LID
#
# Paper (Section 3.3 + Appendix B):
#   "For romanized text, we observed that linear classifiers
#    do not perform very well. Hence, we also experimented
#    with models having larger capacity."
#
#   Model: IndicBERT-v2 (ai4bharat/IndicBERTv2-MLM-only)
#   Why IndicBERT over XLM-R / MuRIL?
#     • IndicBERT supports 24 Indian languages (most coverage)
#     • MuRIL only supports 17 languages
#     • IndicBERT and MuRIL perform similarly, IndicBERT preferred
#       for its wider coverage (Table 7)
#
#   Fine-tuning strategy: Unfreeze ONLY the last 1 transformer layer
#     • Table 8 shows unfreeze-layer-1 beats unfreezing more layers
#     • Unfreezing too many layers leads to overfitting on syn data
#
#   Training data: Synthetic romanized data (from Step 3)
#
#   Results (Table 4):
#     IndicLID-BERT: 80.04% accuracy, 3 sentences/second, ~1.1 GB
#
# Run:  python 05_train_indicbert.py
# ==============================================================

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Silence TensorFlow warnings before importing transformers
os.environ["USE_TF"]             = "0"
os.environ["USE_TORCH"]          = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    INDICBERT_MODEL_NAME,
    BERT_MODEL_DIR,
    ROMANIZED_TRAIN_CSV,
    ROMANIZED_TEST_CSV,
    BENCHMARK_ROMANIZED_CSV,
    INDICBERT_TRAINING_ARGS,
    INDICBERT_NUM_UNFREEZE_LAYERS,
    MAX_TOKEN_LENGTH,
    MODELS_DIR,
    LOGS_DIR,
    NUM_LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    ROMANIZED_LABEL_IDS,
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
# DATASET
# ─────────────────────────────────────────────────────────────────

class RomanizedDataset(Dataset):
    """
    PyTorch Dataset for romanized text classification.
    Wraps the romanized_train.csv / romanized_test.csv.

    Columns expected: sentence, label, label_str, iso
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_TOKEN_LENGTH):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        sentence = str(row["sentence"])
        label    = int(row["label"])

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
# LAYER FREEZING (Appendix B replication)
# ─────────────────────────────────────────────────────────────────

def freeze_layers_except_last_n(model, n_unfreeze: int = 1):
    """
    Freeze all transformer layers except the last n_unfreeze layers.
    Matches the paper's finding that unfreeze-layer-1 is optimal.

    Paper (Table 8):
        unfreeze-layer-1  → 80.04% accuracy  ← BEST
        unfreeze-layer-2  → 79.55%
        unfreeze-layer-4  → 79.47%
        unfreeze-layer-6  → 77.08%
        unfreeze-layer-8  → 76.04%
        unfreeze-layer-11 → 79.88%
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Get all encoder layers
    # IndicBERT uses Albert architecture: encoder.albert_layer_groups
    # or standard BERT architecture: encoder.layer
    encoder = None
    try:
        # Albert architecture (IndicBERT-v2 is Albert-based)
        encoder_layers = model.albert.encoder.albert_layer_groups
    except AttributeError:
        try:
            # Standard BERT / RoBERTa
            encoder_layers = model.bert.encoder.layer
        except AttributeError:
            try:
                encoder_layers = model.roberta.encoder.layer
            except AttributeError:
                encoder_layers = None

    if encoder_layers is not None:
        total_layers = len(encoder_layers)
        # Unfreeze last n_unfreeze layers
        for layer in encoder_layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Unfroze last {n_unfreeze} of {total_layers} encoder layers.")
    else:
        print("  WARNING: Could not identify encoder layers. Unfreezing all.")
        for param in model.parameters():
            param.requires_grad = True

    # Always unfreeze the classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    if hasattr(model, "pooler"):
        for param in model.pooler.parameters():
            param.requires_grad = True

    # Log trainable params
    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params   : {total:,}")
    print(f"  Trainable params: {active:,} ({100*active/total:.1f}%)")


# ─────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────

_accuracy_metric = evaluate.load("accuracy")
_f1_metric       = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = _accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = _f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}


# ─────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────

def train_indicbert(
    train_csv: str = ROMANIZED_TRAIN_CSV,
    test_csv:  str = ROMANIZED_TEST_CSV,
    model_dir: str = BERT_MODEL_DIR,
    n_unfreeze_layers: int = INDICBERT_NUM_UNFREEZE_LAYERS,
):
    """
    Fine-tune IndicBERT-v2 for romanized text language identification.

    Strategy (Appendix B):
    1. Load pretrained IndicBERT-v2
    2. Add softmax classification head (NUM_LABELS classes)
    3. Freeze all layers except last n_unfreeze_layers
    4. Fine-tune on synthetic romanized training data
    """
    print(f"\n{'='*65}")
    print(f"  Training IndicLID-BERT")
    print(f"{'='*65}")

    # ── Validate inputs ──
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Romanized train CSV not found: {train_csv}\n"
            "Run 03_generate_synthetic_romanized.py first."
        )

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR,  exist_ok=True)

    # ── 1. Load tokenizer ──
    print(f"\n[1/6] Loading tokenizer: {INDICBERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)

    # ── 2. Load data ──
    print("[2/6] Loading datasets …")
    train_df = pd.read_csv(train_csv, dtype={"sentence": str, "label": int}).dropna()
    test_df  = pd.read_csv(test_csv,  dtype={"sentence": str, "label": int}).dropna() \
               if os.path.exists(test_csv) else pd.DataFrame(columns=["sentence","label"])

    # Only keep romanized labels (IDs 24-44) for the BERT model
    train_df = train_df[train_df["label"].isin(list(ROMANIZED_LABEL_IDS))].copy()
    test_df  = test_df[test_df["label"].isin(list(ROMANIZED_LABEL_IDS))].copy()

    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test  rows : {len(test_df):,}")
    print(f"  Num labels : {NUM_LABELS}")

    # Remap romanized labels to 0-based for this model
    # (IndicBERT is trained ONLY on romanized text: labels 24-44)
    roman_labels_sorted = sorted(ROMANIZED_LABEL_IDS)
    roman_to_local = {g: i for i, g in enumerate(roman_labels_sorted)}
    local_to_roman = {i: g for g, i in roman_to_local.items()}

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["label"] = train_df["label"].map(roman_to_local)
    test_df["label"]  = test_df["label"].map(roman_to_local)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"]  = test_df["label"].astype(int)

    n_local_labels = len(roman_labels_sorted)
    id2label = {i: ID_TO_LABEL[g] for i, g in local_to_roman.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Save the label mapping for later use by the pipeline
    import json
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(mapping_path, "w") as mf:
        json.dump({
            "id2label": {str(k): v for k, v in id2label.items()},
            "roman_to_local": {str(k): v for k, v in roman_to_local.items()},
            "local_to_roman": {str(k): v for k, v in local_to_roman.items()},
        }, mf, indent=2)

    train_dataset = RomanizedDataset(train_df, tokenizer)
    test_dataset  = RomanizedDataset(test_df,  tokenizer) if not test_df.empty else None

    # ── 3. Load model ──
    print(f"\n[3/6] Loading model: {INDICBERT_MODEL_NAME}")
    print(f"      Classification head: {n_local_labels} romanized classes")
    model = AutoModelForSequenceClassification.from_pretrained(
        INDICBERT_MODEL_NAME,
        num_labels=n_local_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── 4. Freeze layers ──
    print(f"\n[4/6] Freezing layers (keeping last {n_unfreeze_layers} unfrozen) …")
    freeze_layers_except_last_n(model, n_unfreeze=n_unfreeze_layers)

    # ── 5. Training arguments ──
    print("\n[5/6] Setting up training …")
    ta = INDICBERT_TRAINING_ARGS.copy()
    grad_accum = ta.get("gradient_accumulation_steps", 16)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=ta["num_train_epochs"],
        learning_rate=ta["learning_rate"],
        per_device_train_batch_size=ta["per_device_train_batch_size"],
        per_device_eval_batch_size=ta["per_device_eval_batch_size"],
        gradient_accumulation_steps=grad_accum,
        warmup_steps=ta["warmup_steps"],
        weight_decay=ta["weight_decay"],
        logging_steps=200,
        eval_strategy=ta.get("eval_strategy", ta.get("evaluation_strategy", "epoch")),
        save_strategy=ta.get("save_strategy", "epoch"),
        load_best_model_at_end=ta["load_best_model_at_end"],
        metric_for_best_model=ta["metric_for_best_model"],
        fp16=ta["fp16"],
        gradient_checkpointing=ta.get("gradient_checkpointing", False),
        dataloader_num_workers=ta["dataloader_num_workers"],
        report_to=ta["report_to"],
    )

    print(f"  Model      : {INDICBERT_MODEL_NAME}")
    print(f"  Epochs     : {training_args.num_train_epochs}")
    print(f"  LR         : {training_args.learning_rate}")
    print(f"  Eff. batch : {training_args.per_device_train_batch_size * grad_accum}")
    print(f"  FP16       : {training_args.fp16}")
    print(f"  Unfreeze   : {n_unfreeze_layers} layer(s)")

    # ── 6. Train ──
    print("\n[6/6] Starting training …\n")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Save ──
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\n  Model saved to: {model_dir}")

    # ── Evaluate ──
    if test_dataset:
        print("\n  Running final evaluation …")
        results = trainer.evaluate(test_dataset)
        print("\n  ── Final Results (IndicLID-BERT) ──")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"    {k:40s} : {v*100:.2f}%")
            else:
                print(f"    {k}: {v}")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# LAYER UNFREEZING EXPERIMENT (Appendix B)
# ─────────────────────────────────────────────────────────────────

def run_layer_unfreeze_experiment():
    """
    Replicate Table 8 from the paper: test different numbers of
    unfrozen layers to verify n=1 is optimal.

    Trains multiple models (WARNING: very slow).
    """
    print("\n  Layer unfreezing experiment (Table 8 replication) …")
    print("  WARNING: This trains 6 separate models. Very slow.\n")

    n_layers_to_test = [1, 2, 4, 6, 8, 11]
    results = {}

    for n in n_layers_to_test:
        out_dir = os.path.join(MODELS_DIR, f"indiclid_bert_unfreeze{n}")
        print(f"\n  Unfreeze {n} layers → {out_dir}")
        model, tok = train_indicbert(
            model_dir=out_dir,
            n_unfreeze_layers=n,
        )
        results[n] = {"dir": out_dir}

    print("\n  Experiment complete. Compare metrics in:")
    for n, r in results.items():
        print(f"    unfreeze-{n}: {r['dir']}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fine-tune IndicBERT for romanized language identification"
    )
    parser.add_argument(
        "--unfreeze_layers", type=int, default=INDICBERT_NUM_UNFREEZE_LAYERS,
        help=f"Number of transformer layers to unfreeze (default: {INDICBERT_NUM_UNFREEZE_LAYERS})"
    )
    parser.add_argument(
        "--layer_experiment", action="store_true",
        help="Run layer-unfreezing experiment (replicates Table 8)"
    )
    args = parser.parse_args()

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  IndicLID — Step 5: Fine-tune IndicBERT (Romanized LID)  ║")
    print("╚════════════════════════════════════════════════════════════╝")

    if args.layer_experiment:
        run_layer_unfreeze_experiment()
    else:
        train_indicbert(n_unfreeze_layers=args.unfreeze_layers)

    print(f"\n  Next step: python 06_pipeline.py")
