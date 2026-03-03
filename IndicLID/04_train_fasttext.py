# ==============================================================
# 04_train_fasttext.py
#
# STEP 4 — Train IndicLID-FTN and IndicLID-FTR (FastText)
#
# Paper (Section 3.2 + Appendix A):
#   "Linear classifiers using character n-gram features are
#    widely used for LIDs. We use FastText to train our fast,
#    linear classifier."
#
#   IndicLID-FTN: FastText native-script model
#   IndicLID-FTR: FastText romanized-script model
#
#   Both use:
#     • 8-dimension word vectors (Table 6: beyond 8-dim, no gain)
#     • Character n-grams (minn=2, maxn=6)
#     • Supervised classification (loss=softmax)
#
#   Results (Table 3 & 4):
#     IndicLID-FTN: 98.55% accuracy, 30,303 sentences/sec, 318MB
#     IndicLID-FTR: 71.49% accuracy, 37,037 sentences/sec, 357MB
#
# Run:  python 04_train_fasttext.py
#       python 04_train_fasttext.py --model native
#       python 04_train_fasttext.py --model romanized
#       python 04_train_fasttext.py --model both
# ==============================================================

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NATIVE_TRAIN_FT_TXT,
    NATIVE_TEST_FT_TXT,
    ROMANIZED_TRAIN_FT_TXT,
    ROMANIZED_TEST_FT_TXT,
    FTN_MODEL_PATH,
    FTR_MODEL_PATH,
    MODELS_DIR,
    FASTTEXT_PARAMS,
)

# ─────────────────────────────────────────────────────────────────
# FASTTEXT IMPORT
# ─────────────────────────────────────────────────────────────────

def import_fasttext():
    """Import fasttext, trying fasttext-wheel on Windows."""
    try:
        import fasttext
        return fasttext
    except ImportError:
        pass
    try:
        import fasttext as fasttext
        return fasttext
    except ImportError:
        print("  ERROR: fasttext not installed.")
        print("  On Windows: pip install fasttext-wheel")
        print("  On Linux  : pip install fasttext")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# TRAIN ONE FASTTEXT MODEL
# ─────────────────────────────────────────────────────────────────

def train_fasttext_model(
    train_file: str,
    test_file:  str,
    model_path: str,
    model_name: str,
    params:     dict = None,
) -> dict:
    """
    Train a FastText supervised classifier.

    Parameters
    ----------
    train_file  : path to FastText-format train .txt file
    test_file   : path to FastText-format test .txt file
    model_path  : where to save the .bin model
    model_name  : display name (FTN / FTR) for logging
    params      : FastText hyperparameters dict

    Returns dict with evaluation metrics.
    """
    ft = import_fasttext()

    if not os.path.exists(train_file):
        print(f"  ERROR: Train file not found: {train_file}")
        print(f"  Run the preprocessing steps first.")
        return {}

    if params is None:
        params = FASTTEXT_PARAMS.copy()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training IndicLID-{model_name}")
    print(f"{'='*60}")
    print(f"  Train file : {train_file}")
    print(f"  Model path : {model_path}")
    print(f"  Parameters :")
    for k, v in params.items():
        print(f"    {k:15s} = {v}")
    print()

    # Count train samples
    with open(train_file, encoding="utf-8", errors="replace") as f:
        n_train = sum(1 for line in f if line.strip())
    print(f"  Training on {n_train:,} examples …")

    start_time = time.time()

    # Train the model
    model = ft.train_supervised(
        input=train_file,
        dim=params.get("dim", 8),
        epoch=params.get("epoch", 25),
        lr=params.get("lr", 0.5),
        wordNgrams=params.get("wordNgrams", 2),
        minn=params.get("minn", 2),
        maxn=params.get("maxn", 6),
        minCount=params.get("minCount", 1),
        loss=params.get("loss", "softmax"),
        thread=params.get("thread", 4),
        verbose=params.get("verbose", 2),
    )

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")

    # ── Save model ──
    model.save_model(model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model saved: {model_path}  ({model_size_mb:.1f} MB)")

    # ── Evaluate on test set ──
    metrics = {}
    if os.path.exists(test_file):
        print(f"\n  Evaluating on test set: {test_file}")
        with open(test_file, encoding="utf-8", errors="replace") as f:
            n_test = sum(1 for line in f if line.strip())
        print(f"  Test set: {n_test:,} examples")

        # FastText evaluate returns (n_samples, precision, recall)
        n, precision, recall = model.test(test_file)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        metrics = {
            "n_test":    n,
            "precision": round(precision * 100, 2),
            "recall":    round(recall    * 100, 2),
            "f1":        round(f1        * 100, 2),
        }

        print(f"\n  ── Results (IndicLID-{model_name}) ──")
        print(f"    N            : {n:,}")
        print(f"    Precision@1  : {precision*100:.2f}%")
        print(f"    Recall@1     : {recall*100:.2f}%")
        print(f"    F1           : {f1*100:.2f}%")

        # Detailed per-class evaluation
        print_per_class_results(model, test_file, model_name)

    # ── Throughput benchmark ──
    print(f"\n  Measuring throughput …")
    sample_sentences = [
        "नमस्ते आप कैसे हैं",  # Hindi
        "ami bhalo achi",        # Bengali romanized
        "vanakkam naan nalama",  # Tamil romanized
    ]
    t0 = time.time()
    N_ITERS = 10_000
    for _ in range(N_ITERS):
        for s in sample_sentences:
            model.predict(s, k=1)
    t1 = time.time()
    throughput = int(N_ITERS * len(sample_sentences) / (t1 - t0))
    print(f"  Throughput: ~{throughput:,} sentences/second")
    metrics["throughput"] = throughput

    return metrics


# ─────────────────────────────────────────────────────────────────
# PER-CLASS DETAIL
# ─────────────────────────────────────────────────────────────────

def print_per_class_results(model, test_file: str, model_name: str):
    """Print per-class precision/recall for the FastText model."""
    print(f"\n  Per-class precision/recall (IndicLID-{model_name}):")

    # FastText's test_label returns dict: label → (precision, recall, count)
    try:
        result = model.test_label(test_file)
    except Exception as e:
        print(f"  (Could not get per-class results: {e})")
        return

    if not result:
        print("  (No per-class results available)")
        return

    # Sort by label name
    sorted_labels = sorted(result.keys())
    print(f"    {'Label':25s}  {'P':>8s}  {'R':>8s}  {'Count':>8s}")
    print(f"    {'─'*55}")
    for label in sorted_labels:
        v = result[label]
        p = v.get("precision", 0) * 100
        r = v.get("recall", 0) * 100
        n = v.get("count", 0)
        label_clean = label.replace("__label__", "")
        print(f"    {label_clean:25s}  {p:8.2f}  {r:8.2f}  {n:8,}")


# ─────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH (Appendix A replication)
# ─────────────────────────────────────────────────────────────────

def run_dimension_search(train_file: str, test_file: str, model_name: str):
    """
    Replicate Table 6 from the paper:
    Test model accuracy vs. dimension (4, 8, 16, 32, 64, 128, ...).
    Paper found: 8-dim is the optimal trade-off.
    """
    ft = import_fasttext()
    dims_to_test = [4, 8, 16, 32, 64, 128]

    print(f"\n{'='*65}")
    print(f"  Hyperparameter search: dimension for IndicLID-{model_name}")
    print(f"  Replicating Table 6 from the paper …")
    print(f"{'='*65}")
    print(f"  {'Dim':>8}  {'Precision%':>12}  {'Recall%':>10}  {'F1%':>8}  {'Size(MB)':>10}")
    print(f"  {'─'*55}")

    results = []
    for dim in dims_to_test:
        params = FASTTEXT_PARAMS.copy()
        params["dim"] = dim
        params["verbose"] = 0   # silent for sweep

        model = ft.train_supervised(
            input=train_file,
            dim=dim,
            epoch=params["epoch"],
            lr=params["lr"],
            wordNgrams=params["wordNgrams"],
            minn=params["minn"],
            maxn=params["maxn"],
            minCount=params["minCount"],
            loss=params["loss"],
            thread=params["thread"],
            verbose=0,
        )

        _, p, r = model.test(test_file)
        f1 = 2*p*r / (p+r+1e-9)

        # Temp save to measure size
        tmp_path = f"/tmp/_ft_dim{dim}.bin"
        model.save_model(tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024*1024)

        print(f"  {dim:>8}  {p*100:>11.2f}%  {r*100:>9.2f}%  {f1*100:>7.2f}%  {size_mb:>9.1f}M")
        results.append({"dim": dim, "precision": p, "recall": r, "f1": f1,
                        "size_mb": size_mb})

    return results


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train IndicLID FastText models (FTN and/or FTR)"
    )
    parser.add_argument(
        "--model", choices=["native", "romanized", "both"],
        default="both",
        help="Which model to train (default: both)"
    )
    parser.add_argument(
        "--dim_search", action="store_true",
        help="Run dimension hyperparameter search (replicates Table 6)"
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  IndicLID — Step 4: Train FastText Models (FTN + FTR)  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    all_metrics = {}

    # ── Train Native (IndicLID-FTN) ──
    if args.model in ("native", "both"):
        if args.dim_search:
            run_dimension_search(NATIVE_TRAIN_FT_TXT, NATIVE_TEST_FT_TXT, "FTN")

        metrics_ftn = train_fasttext_model(
            train_file=NATIVE_TRAIN_FT_TXT,
            test_file=NATIVE_TEST_FT_TXT,
            model_path=FTN_MODEL_PATH,
            model_name="FTN",
        )
        all_metrics["FTN"] = metrics_ftn

    # ── Train Romanized (IndicLID-FTR) ──
    if args.model in ("romanized", "both"):
        if args.dim_search:
            run_dimension_search(ROMANIZED_TRAIN_FT_TXT, ROMANIZED_TEST_FT_TXT, "FTR")

        metrics_ftr = train_fasttext_model(
            train_file=ROMANIZED_TRAIN_FT_TXT,
            test_file=ROMANIZED_TEST_FT_TXT,
            model_path=FTR_MODEL_PATH,
            model_name="FTR",
        )
        all_metrics["FTR"] = metrics_ftr

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  Training Summary")
    print(f"{'='*60}")
    for model_name, m in all_metrics.items():
        if m:
            print(f"\n  IndicLID-{model_name}:")
            print(f"    Precision  : {m.get('precision','—')}%")
            print(f"    Recall     : {m.get('recall','—')}%")
            print(f"    F1         : {m.get('f1','—')}%")
            print(f"    Throughput : ~{m.get('throughput','—'):,} sent/s")

    print(f"\n  Models saved to: {MODELS_DIR}")
    print(f"  Next step: python 05_train_indicbert.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
