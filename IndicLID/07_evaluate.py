# ==============================================================
# 07_evaluate.py
#
# STEP 7 — Evaluate IndicLID on the Bhasha-Abhijnaanam Benchmark
#
# Replicates the following paper tables / figures:
#
#   Table 3 : FTN results (per-language, native-script)
#   Table 4 : Ensemble results (per-language, romanized)
#   Table 5 : Synthetic vs original romanized comparison
#   Table 6 : FastText dimension search (run 04_train_fasttext.py first)
#   Table 8 : BERT layer unfreeze experiment (run 05_train_indicbert.py)
#   Table 9 : Confidence threshold sweep
#   Figure 2: Confusion matrices (native + romanized)
#   Figure 3: Accuracy vs word-count buckets
#
# Usage:
#   python 07_evaluate.py                    # full evaluation
#   python 07_evaluate.py --mode native      # native path only
#   python 07_evaluate.py --mode romanized   # romanized path only
#   python 07_evaluate.py --mode both        # both (default)
#   python 07_evaluate.py --no_bert          # skip BERT (faster)
#   python 07_evaluate.py --plot             # save confusion matrix PNGs
# ==============================================================

import os
import sys
import json
import csv
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BENCHMARK_NATIVE_CSV   as BENCHMARK_NATIVE_PATH,
    BENCHMARK_ROMANIZED_CSV as BENCHMARK_ROMAN_PATH,
    NATIVE_TEST_CSV,
    ROMANIZED_TEST_CSV     as ROMAN_TEST_CSV,
    FTN_MODEL_PATH,
    FTR_MODEL_PATH,
    BERT_MODEL_DIR,
    CONFIDENCE_THRESHOLD,
    ROMAN_CHAR_THRESHOLD,
    LABEL_TO_ID,
    ID_TO_LABEL,
    NATIVE_LABEL_IDS,
    ROMANIZED_LABEL_IDS,
    ENGLISH_LABEL_ID,
    OTHERS_LABEL_ID,
    INDIC_LANGUAGES,
)


# ─────────────────────────────────────────────────────────────────
# METRICS HELPERS
# ─────────────────────────────────────────────────────────────────

def compute_metrics_from_lists(
    ground_truths: list, predictions: list, label_set: Optional[list] = None
) -> dict:
    """
    Compute accuracy, macro-P, macro-R, macro-F1 and per-label F1.
    Uses sklearn for correctness.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    acc = accuracy_score(ground_truths, predictions)
    p   = precision_score(ground_truths, predictions, average="macro", zero_division=0)
    r   = recall_score(ground_truths, predictions,    average="macro", zero_division=0)
    f1  = f1_score(ground_truths, predictions,        average="macro", zero_division=0)

    label_list = sorted(set(ground_truths) | set(predictions)) if label_set is None else label_set
    report = classification_report(
        ground_truths, predictions,
        labels=label_list, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(ground_truths, predictions, labels=label_list)

    return {
        "accuracy": acc,
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "per_label": report,
        "labels": label_list,
        "confusion_matrix": cm,
    }


def print_summary_table(metrics: dict, title: str):
    """Print a compact results table similar to the paper."""
    print(f"\n  ── {title} ──")
    print(f"  Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"  Macro Precision:  {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall:     {metrics['macro_recall']*100:.2f}%")
    print(f"  Macro F1:         {metrics['macro_f1']*100:.2f}%")


def print_per_label_table(metrics: dict, top_n: int = 30):
    """Print per-label P/R/F1 (mirrors Table 3 / Table 4 in the paper)."""
    print(f"\n  {'Label':20s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*62}")

    per_label = metrics["per_label"]
    rows = []
    for label in metrics["labels"]:
        if label in per_label:
            stat = per_label[label]
            rows.append((
                label,
                stat["precision"],
                stat["recall"],
                stat["f1-score"],
                int(stat["support"]),
            ))

    # Sort by support (descending)
    rows.sort(key=lambda x: x[4], reverse=True)
    for row in rows[:top_n]:
        print(f"  {row[0]:20s} {row[1]*100:>9.1f}% {row[2]*100:>9.1f}% {row[3]*100:>9.1f}% {row[4]:>10d}")


# ─────────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray, labels: list, title: str, output_path: str
):
    """Save a confusion matrix PNG (Figure 2 style)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels))))
        sns.heatmap(
            cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        print("  matplotlib/seaborn not installed. Skipping confusion matrix plot.")


# ─────────────────────────────────────────────────────────────────
# WORD-COUNT BUCKET ANALYSIS (Figure 3)
# ─────────────────────────────────────────────────────────────────

def accuracy_by_word_count(
    texts: list, ground_truths: list, predictions: list
) -> dict:
    """
    Compute accuracy per word-count bucket (replicates Figure 3).
    Buckets: 1, 2, 3, 4, 5-9, 10-19, 20+
    """
    BUCKETS = [
        (1,  1,  "1"),
        (2,  2,  "2"),
        (3,  3,  "3"),
        (4,  4,  "4"),
        (5,  9,  "5-9"),
        (10, 19, "10-19"),
        (20, 9999, "20+"),
    ]

    results = {}
    for lo, hi, name in BUCKETS:
        correct = total = 0
        for text, gt, pred in zip(texts, ground_truths, predictions):
            wc = len(text.split())
            if lo <= wc <= hi:
                total   += 1
                correct += (gt == pred)
        results[name] = (correct / total * 100) if total > 0 else 0.0
    return results


def print_word_count_accuracy(wc_acc: dict, title: str):
    """Print word-count bucket accuracy table (Figure 3)."""
    print(f"\n  ── {title}: Accuracy by Word Count (Figure 3) ──")
    print(f"  {'Words':>8}  {'Accuracy':>10}")
    print(f"  {'─'*22}")
    for bucket, acc in wc_acc.items():
        print(f"  {bucket:>8s}  {acc:>9.1f}%")


# ─────────────────────────────────────────────────────────────────
# LOAD BENCHMARK
# ─────────────────────────────────────────────────────────────────

def load_csv_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        print(f"  Run 01_download_datasets.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from: {path}")
    return df


# ─────────────────────────────────────────────────────────────────
# EVALUATE FASTTEXT (FTN / FTR)
# ─────────────────────────────────────────────────────────────────

def evaluate_fasttext(
    model_path: str, df: pd.DataFrame, model_name: str, plot: bool = False
) -> dict:
    """
    Evaluate a FastText model on a DataFrame with 'text' and 'label' columns.
    """
    if df.empty:
        print(f"  [{model_name}] No data. Skipping.")
        return {}

    try:
        import fasttext
    except ImportError:
        try:
            import fasttext_wheel as fasttext
        except ImportError:
            print(f"  [{model_name}] fasttext not installed.")
            return {}

    if not os.path.exists(model_path):
        print(f"  [{model_name}] Model not found: {model_path}")
        return {}

    print(f"\n  [{model_name}] Loading model: {model_path}")
    model = fasttext.load_model(model_path)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    preds = []
    t0    = time.time()
    for text in texts:
        pred_labels, pred_probs = model.predict(text, k=1)
        pred = pred_labels[0].replace("__label__", "")
        preds.append(pred)
    elapsed = time.time() - t0
    throughput = len(texts) / elapsed if elapsed > 0 else 0

    print(f"  [{model_name}] Predicted {len(texts)} samples in {elapsed:.1f}s "
          f"({throughput:.0f} sent/s)")

    metrics = compute_metrics_from_lists(labels, preds)
    print_summary_table(metrics, f"{model_name} results")
    print_per_label_table(metrics)

    # Word-count analysis
    wc_acc = accuracy_by_word_count(texts, labels, preds)
    print_word_count_accuracy(wc_acc, model_name)

    if plot:
        out_dir = Path(model_path).parent
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["labels"],
            f"{model_name} Confusion Matrix",
            str(out_dir / f"{model_name.lower()}_confusion.png"),
        )

    metrics["throughput"] = throughput
    return metrics


# ─────────────────────────────────────────────────────────────────
# EVALUATE BERT STANDALONE
# ─────────────────────────────────────────────────────────────────

def evaluate_bert(
    df: pd.DataFrame, model_dir: str, plot: bool = False
) -> dict:
    """
    Evaluate IndicLID-BERT on romanized test set.
    """
    if df.empty:
        print("  [BERT] No data. Skipping.")
        return {}

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("  [BERT] transformers not installed.")
        return {}

    if not os.path.isdir(model_dir):
        print(f"  [BERT] Model directory not found: {model_dir}")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label mapping
    label_map_path = os.path.join(model_dir, "label_mapping.json")
    if not os.path.exists(label_map_path):
        print(f"  [BERT] label_mapping.json not found in {model_dir}")
        return {}
    with open(label_map_path) as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    BATCH = 64
    preds = []
    t0    = time.time()

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        inputs = tokenizer(
            batch, truncation=True, max_length=128,
            padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        batch_preds = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(id2label.get(p, "xx") for p in batch_preds)

    elapsed    = time.time() - t0
    throughput = len(texts) / elapsed if elapsed > 0 else 0
    print(f"\n  [BERT] Predicted {len(texts)} samples in {elapsed:.1f}s "
          f"({throughput:.0f} sent/s)")

    metrics = compute_metrics_from_lists(labels, preds)
    print_summary_table(metrics, "IndicLID-BERT (romanized) results")
    print_per_label_table(metrics)

    if plot:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            metrics["labels"],
            "IndicLID-BERT Confusion Matrix",
            os.path.join(model_dir, "bert_confusion.png"),
        )

    metrics["throughput"] = throughput
    return metrics


# ─────────────────────────────────────────────────────────────────
# EVALUATE PIPELINE (ENSEMBLE)
# ─────────────────────────────────────────────────────────────────

def evaluate_pipeline(
    df: pd.DataFrame,
    pipeline,
    thresholds: Optional[list] = None,
    plot: bool = False,
) -> dict:
    """
    Evaluate the IndicLID ensemble pipeline on a romanized test set.
    Optionally sweep confidence thresholds (Table 9).
    """
    if df.empty:
        print("  [Pipeline] No data.")
        return {}

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    if thresholds is None:
        thresholds = [CONFIDENCE_THRESHOLD]

    all_results = {}

    for thr in thresholds:
        pipeline.confidence_threshold = thr
        preds = []
        bert_calls = 0
        t0 = time.time()

        for text in texts:
            result = pipeline.identify(text)
            preds.append(result["label"])
            if result["model_used"] == "BERT":
                bert_calls += 1

        elapsed    = time.time() - t0
        throughput = len(texts) / elapsed if elapsed > 0 else 0

        print(f"\n  [Pipeline] threshold={thr:.1f}  "
              f"BERT calls={bert_calls}/{len(texts)} ({bert_calls/len(texts)*100:.1f}%)  "
              f"speed={throughput:.0f} sent/s")

        metrics = compute_metrics_from_lists(labels, preds)
        print_summary_table(metrics, f"Ensemble (threshold={thr})")

        if thr == CONFIDENCE_THRESHOLD:
            print_per_label_table(metrics)
            wc_acc = accuracy_by_word_count(texts, labels, preds)
            print_word_count_accuracy(wc_acc, "Ensemble")

            if plot:
                plot_confusion_matrix(
                    metrics["confusion_matrix"],
                    metrics["labels"],
                    f"IndicLID Ensemble (threshold={thr})",
                    os.path.join(BERT_MODEL_DIR, "pipeline_confusion.png"),
                )

        metrics["throughput"] = throughput
        metrics["bert_calls"] = bert_calls
        all_results[thr] = metrics

    # Print threshold sweep summary (Table 9)
    if len(thresholds) > 1:
        print(f"\n  ── Table 9: Threshold Sweep ──")
        print(f"  {'Thresh':>8}  {'Accuracy':>10}  {'Macro-F1':>10}  {'Speed(s/s)':>12}  {'BERT%':>8}")
        print(f"  {'─'*55}")
        for thr, m in sorted(all_results.items()):
            bpct = m["bert_calls"] / len(texts) * 100
            print(f"  {thr:>8.1f}  {m['accuracy']*100:>9.2f}%  {m['macro_f1']*100:>9.2f}%  "
                  f"{m['throughput']:>12.0f}  {bpct:>7.1f}%")

    return all_results


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IndicLID on the Bhasha-Abhijnaanam benchmark"
    )
    parser.add_argument("--mode", choices=["native", "romanized", "both"],
                        default="both",
                        help="Evaluation mode (default: both)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Use Bhasha-Abhijnaanam benchmark instead of own test split")
    parser.add_argument("--no_bert", action="store_true",
                        help="Skip BERT evaluation (faster)")
    parser.add_argument("--threshold_sweep", action="store_true",
                        help="Sweep confidence thresholds (Table 9)")
    parser.add_argument("--plot", action="store_true",
                        help="Save confusion matrix PNGs")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  IndicLID — Evaluation (ACL 2023, AI4Bharat)           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Choose data source ──
    if args.benchmark:
        native_path    = BENCHMARK_NATIVE_PATH
        romanized_path = BENCHMARK_ROMAN_PATH
        data_label     = "Bhasha-Abhijnaanam Benchmark"
    else:
        native_path    = NATIVE_TEST_CSV
        romanized_path = ROMAN_TEST_CSV
        data_label     = "Internal Test Split"

    print(f"  Data source: {data_label}")

    # ── 1. NATIVE evaluation ──
    native_metrics = {}
    if args.mode in ("native", "both"):
        print(f"\n  Loading native test data …")
        df_native = load_csv_dataset(native_path)
        if not df_native.empty:
            native_metrics = evaluate_fasttext(
                FTN_MODEL_PATH, df_native,
                model_name="IndicLID-FTN",
                plot=args.plot
            )

    # ── 2. ROMANIZED evaluation ──
    if args.mode in ("romanized", "both"):
        print(f"\n  Loading romanized test data …")
        df_roman = load_csv_dataset(romanized_path)

        if not df_roman.empty:
            # ── 2a. FTR alone ──
            print("\n  ── Evaluating IndicLID-FTR (alone) ──")
            ftr_metrics = evaluate_fasttext(
                FTR_MODEL_PATH, df_roman,
                model_name="IndicLID-FTR",
                plot=args.plot
            )

            # ── 2b. BERT alone ──
            if not args.no_bert:
                print("\n  ── Evaluating IndicLID-BERT (alone) ──")
                bert_metrics = evaluate_bert(
                    df_roman, BERT_MODEL_DIR, plot=args.plot
                )

            # ── 2c. Ensemble Pipeline ──
            print("\n  ── Evaluating IndicLID Ensemble Pipeline ──")
            from module_06_pipeline import IndicLIDPipeline  # noqa: local import
            pipeline = IndicLIDPipeline(use_bert=not args.no_bert)

            thresholds = (
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                if args.threshold_sweep
                else [CONFIDENCE_THRESHOLD]
            )
            pipeline_results = evaluate_pipeline(
                df_roman, pipeline,
                thresholds=thresholds,
                plot=args.plot
            )

    # ── 3. Comparison summary ──
    print(f"\n  ─────────────────────────────────────────────────────────")
    print(f"  SUMMARY (paper Table 3, Table 4 references)")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Paper results (for reference):")
    print(f"    IndicLID-FTN  (native)     : 98.55% accuracy")
    print(f"    IndicLID-FTR  (romanized)  : 71.49% accuracy")
    print(f"    IndicLID-BERT (romanized)  : 80.04% accuracy")
    print(f"    IndicLID Ens. (threshold=0.6): 80.40% accuracy, ~10 sent/s")

    if "accuracy" in native_metrics:
        print(f"\n  Our IndicLID-FTN: {native_metrics['accuracy']*100:.2f}% accuracy")
    print(f"\n  Evaluation complete.")
    print(f"  Run 08_inference.py for interactive use.")


# ── Local import alias for the pipeline (avoids the '06_' prefix issue) ──
import importlib.util as _ilu
import types as _types

def _import_pipeline():
    """
    Dynamically import 06_pipeline.py despite its numeric prefix.
    Returns the module object.
    """
    spec = _ilu.spec_from_file_location(
        "module_06_pipeline",
        Path(__file__).parent / "06_pipeline.py"
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules["module_06_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-load at module level so the `from module_06_pipeline import …` line works
_import_pipeline()


if __name__ == "__main__":
    main()
