# ==============================================================
# 08_evaluate.py
#
# STEP 8 — Full Evaluation
#
# Reproduces Tables 1 & 2 from the paper:
#   • XLM-RoBERTa Native  vs SVM Native  (Table 1)
#   • XLM-RoBERTa Romanized vs SVM Romanized (Table 2)
#
# Also evaluates the full pipeline on a test file.
#
# Metrics: Precision, Recall, F1-score, Accuracy (per class + overall)
#
# Run:  python 08_evaluate.py --model native
#       python 08_evaluate.py --model romanized
#       python 08_evaluate.py --model both
# ==============================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NATIVE_TEST_CSV,
    ROMANIZED_TEST_CSV,
    NATIVE_MODEL_DIR,
    ROMANIZED_MODEL_DIR,
    LANGUAGE_LABELS,
    MAX_TOKEN_LENGTH,
    PROCESSED_DIR,
)

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
NUM_LABELS     = len(LANGUAGE_LABELS)
LABEL_NAMES    = [LANGUAGE_LABELS[i] for i in range(NUM_LABELS)]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
# ROBERTA EVALUATION
# ─────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text  = str(self.data.loc[idx, "Sentence"])
        label = int(self.data.loc[idx, "Language"])
        enc   = self.tokenizer(
            text, truncation=True, max_length=MAX_TOKEN_LENGTH,
            padding=False, return_tensors=None
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         label,
        }


def collate_fn(batch, tokenizer):
    """Pad batch to max length in the batch."""
    texts  = [b["input_ids"]      for b in batch]
    masks  = [b["attention_mask"] for b in batch]
    labels = [b["labels"]         for b in batch]

    padded = tokenizer.pad(
        {"input_ids": texts, "attention_mask": masks},
        padding=True,
        return_tensors="pt",
    )
    return {**padded, "labels": torch.tensor(labels)}


def evaluate_roberta(
    model_dir: str,
    test_csv: str,
    model_name: str = "RoBERTa",
    batch_size: int = 128,
) -> dict:
    """
    Load a fine-tuned XLM-RoBERTa model and evaluate on test set.
    Returns metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"  Model : {model_dir}")
    print(f"  Data  : {test_csv}")
    print(f"{'='*60}")

    if not os.path.isdir(model_dir):
        print(f"  Model not found. Run training first.  Skipping.")
        return {}

    if not os.path.exists(test_csv):
        print(f"  Test CSV not found: {test_csv}  Skipping.")
        return {}

    # Load
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()

    df         = pd.read_csv(test_csv).dropna()
    dataset    = EvalDataset(df, tokenizer)
    from functools import partial
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    all_preds, all_labels = [], []
    print(f"  Running inference on {len(df):,} samples …")

    for batch in loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(f"\n  Results ({model_name})")
    print(f"  {'Accuracy':20s}: {acc*100:.2f}%")
    print(f"  {'Precision':20s}: {p*100:.2f}%")
    print(f"  {'Recall':20s}: {r*100:.2f}%")
    print(f"  {'F1 (macro)':20s}: {f1*100:.2f}%")

    print(f"\n  Per-class report:")
    print(
        classification_report(
            all_labels, all_preds,
            target_names=LABEL_NAMES[:max(all_labels)+1],
            zero_division=0,
        )
    )

    return {
        "model":     model_name,
        "accuracy":  round(acc * 100, 2),
        "precision": round(p   * 100, 2),
        "recall":    round(r   * 100, 2),
        "f1":        round(f1  * 100, 2),
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ─────────────────────────────────────────────────────────────────
# SVM BASELINE  (corresponds to Table 1 & 2 in the paper)
# ─────────────────────────────────────────────────────────────────

def evaluate_svm(
    train_csv: str,
    test_csv: str,
    feature: str = "count",   # "count" or "tfidf"
    use_kfold: bool = False,
    model_name: str = "SVM",
) -> dict:
    """
    Train a LinearSVC on the training CSV and evaluate on the test CSV.
    Reproduces the baseline SVM results from the paper.

    Parameters
    ----------
    feature   : "count" (Count Vectorizer) or "tfidf" (TF-IDF)
    use_kfold : if True, uses 10-fold stratified CV (paper's native SVM)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} (feature={feature}, kfold={use_kfold})")
    print(f"{'='*60}")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("  CSV files not found. Skipping SVM evaluation.")
        return {}

    train_df = pd.read_csv(train_csv).dropna().sample(
        frac=1, random_state=42
    )
    test_df  = pd.read_csv(test_csv).dropna()

    X_train = train_df["Sentence"].astype(str).tolist()
    y_train = train_df["Language"].tolist()
    X_test  = test_df["Sentence"].astype(str).tolist()
    y_test  = test_df["Language"].tolist()

    # Feature extraction
    if feature == "tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=100_000,
        )
    else:
        vectorizer = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=100_000,
        )

    print("  Fitting vectorizer …")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = LinearSVC(max_iter=5000, C=1.0)

    if use_kfold:
        print("  Running 10-fold Stratified CV …")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        preds = cross_val_predict(clf, X_train_vec, y_train, cv=skf)
        acc = accuracy_score(y_train, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            y_train, preds, average="macro", zero_division=0
        )
        print(f"  CV Accuracy : {acc*100:.2f}%")
    else:
        print("  Training SVM …")
        clf.fit(X_train_vec, y_train)
        preds  = clf.predict(X_test_vec)
        acc    = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )

    print(f"\n  Results ({model_name})")
    print(f"  {'Accuracy':20s}: {acc*100:.2f}%")
    print(f"  {'Precision':20s}: {p*100:.2f}%")
    print(f"  {'Recall':20s}: {r*100:.2f}%")
    print(f"  {'F1 (macro)':20s}: {f1*100:.2f}%")

    return {
        "model":     model_name,
        "accuracy":  round(acc * 100, 2),
        "precision": round(p   * 100, 2),
        "recall":    round(r   * 100, 2),
        "f1":        round(f1  * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str = None,
):
    """Plot and optionally save a confusion matrix heatmap."""
    cm         = confusion_matrix(labels, preds)
    cm_norm    = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n          = cm.shape[0]
    label_strs = [LANGUAGE_LABELS.get(i, str(i)) for i in range(n)]

    plt.figure(figsize=(max(10, n), max(8, n - 1)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=label_strs, yticklabels=label_strs,
        cmap="Blues", vmin=0, vmax=1,
    )
    plt.title(title)
    plt.ylabel("True Language")
    plt.xlabel("Predicted Language")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix saved: {save_path}")
    else:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────────────────────────
# COMPARISON TABLE  (reproduces Tables 1 & 2 from the paper)
# ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: list, title: str):
    """Print a neat comparison table of all models."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Model':40s} {'P':>7} {'R':>7} {'F1':>7} {'Acc':>7}")
    print(f"  {'-'*70}")
    for r in results:
        if r:
            print(
                f"  {r.get('model','?'):40s}"
                f" {r.get('precision',0):7.2f}"
                f" {r.get('recall',0):7.2f}"
                f" {r.get('f1',0):7.2f}"
                f" {r.get('accuracy',0):7.2f}"
            )
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BharatBhasaNet Evaluation"
    )
    parser.add_argument(
        "--model", choices=["native", "romanized", "both"],
        default="both", help="Which model(s) to evaluate"
    )
    parser.add_argument(
        "--skip-svm", action="store_true",
        help="Skip SVM baseline (faster)"
    )
    args = parser.parse_args()

    from config import NATIVE_TRAIN_CSV, ROMANIZED_TRAIN_CSV

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    native_results, roman_results = [], []

    # ── NATIVE ──
    if args.model in ("native", "both"):
        # RoBERTa Native
        r_native = evaluate_roberta(
            model_dir=NATIVE_MODEL_DIR,
            test_csv=NATIVE_TEST_CSV,
            model_name="XLM-RoBERTa Native",
        )
        native_results.append(r_native)

        if r_native.get("preds") is not None:
            plot_confusion_matrix(
                r_native["preds"], r_native["labels"],
                title="XLM-RoBERTa Native Confusion Matrix",
                save_path=os.path.join(PROCESSED_DIR, "cm_native_roberta.png"),
            )

        # SVM baselines (optional)
        if not args.skip_svm:
            r_svm_count = evaluate_svm(
                NATIVE_TRAIN_CSV, NATIVE_TEST_CSV,
                feature="count", use_kfold=True,
                model_name="SVM Native + Count Vectorizer",
            )
            r_svm_tfidf = evaluate_svm(
                NATIVE_TRAIN_CSV, NATIVE_TEST_CSV,
                feature="tfidf", use_kfold=True,
                model_name="SVM Native + TF-IDF Vectorizer",
            )
            native_results.extend([r_svm_count, r_svm_tfidf])

        print_comparison_table(native_results, "Table 1: Native Model Comparison")

    # ── ROMANIZED ──
    if args.model in ("romanized", "both"):
        r_roman = evaluate_roberta(
            model_dir=ROMANIZED_MODEL_DIR,
            test_csv=ROMANIZED_TEST_CSV,
            model_name="XLM-RoBERTa Romanized",
        )
        roman_results.append(r_roman)

        if r_roman.get("preds") is not None:
            plot_confusion_matrix(
                r_roman["preds"], r_roman["labels"],
                title="XLM-RoBERTa Romanized Confusion Matrix",
                save_path=os.path.join(PROCESSED_DIR, "cm_romanized_roberta.png"),
            )

        if not args.skip_svm:
            r_svm_roman = evaluate_svm(
                ROMANIZED_TRAIN_CSV, ROMANIZED_TEST_CSV,
                feature="count", use_kfold=False,
                model_name="SVM Romanized + Count Vectorizer",
            )
            roman_results.append(r_svm_roman)

        print_comparison_table(roman_results, "Table 2: Romanized Model Comparison")

    print("\n✅ Evaluation complete.")
    print("Next step: Run  python 09_inference.py  for interactive demo.")


if __name__ == "__main__":
    main()
