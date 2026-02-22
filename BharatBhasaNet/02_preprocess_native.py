# ==============================================================
# 02_preprocess_native.py
#
# STEP 2 — Preprocess native-script sentences.
#
# What the paper did (Section III-A):
#   • Remove memorable characters, numbers, and punctuation
#   • For non-English languages, exclude English characters
#   • Structure as CSV with columns: "Sentence", "Language"
#   • 600,000 training (50k × 12) + 360,000 testing (30k × 12)
#
# Run:  python 02_preprocess_native.py
# ==============================================================

import os
import re
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_NATIVE_DIR,
    PROCESSED_DIR,
    LANGUAGE_LABELS,
    LANGUAGE_CODES,
    LABEL_TO_ID,
    NATIVE_TRAIN_CSV,
    NATIVE_TEST_CSV,
)


# ─────────────────────────────────────────────────────────────────
# TEXT CLEANING FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def remove_noise(text: str, lang_name: str) -> str:
    """
    Clean a sentence for native-script training.

    Steps (as described in the paper):
    1. Remove numbers (digits)
    2. Remove punctuation and special characters
    3. For non-English languages, remove ASCII/Latin characters
       (English letters) because they would cause leakage
    4. Strip leading/trailing whitespace
    5. Collapse multiple spaces into one
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove digits (0-9 and Unicode digits)
    text = re.sub(r'\d+', '', text)

    # 2. Remove punctuation and special characters
    #    Keep letters and spaces only
    text = re.sub(r'[^\w\s]', '', text)

    # 3. For non-English languages, remove Latin/ASCII letters
    #    (Hindi, Bengali etc. should not contain a, b, c …)
    if lang_name.lower() != "english":
        text = re.sub(r'[a-zA-Z]', '', text)

    # 4. Remove extra underscores left by \w
    text = text.replace('_', '')

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_valid_sentence(text: str, min_chars: int = 5) -> bool:
    """
    Reject sentences that are too short after cleaning
    or completely empty.
    """
    return len(text) >= min_chars


# ─────────────────────────────────────────────────────────────────
# LOAD AND CLEAN ONE LANGUAGE
# ─────────────────────────────────────────────────────────────────

def load_language_file(
    filepath: str,
    lang_name: str,
    lang_label: int,
    n_max: int,
) -> pd.DataFrame:
    """
    Read a raw .txt file (one sentence per line) for one language,
    clean it, and return a DataFrame with columns
    ['Sentence', 'Language'].

    Parameters
    ----------
    filepath   : path to the raw .txt file
    lang_name  : human-readable language name (e.g. 'Hindi')
    lang_label : integer label (e.g. 3)
    n_max      : maximum number of cleaned sentences to keep
    """
    if not os.path.exists(filepath):
        print(f"    FILE NOT FOUND: {filepath}")
        return pd.DataFrame(columns=["Sentence", "Language"])

    rows = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(rows) >= n_max:
                break
            cleaned = remove_noise(line, lang_name)
            if is_valid_sentence(cleaned):
                rows.append({"Sentence": cleaned, "Language": lang_label})

    df = pd.DataFrame(rows)
    print(f"    {lang_name:12s} | label={lang_label:2d} | {len(df):7,} sentences")
    return df


# ─────────────────────────────────────────────────────────────────
# BUILD FULL TRAIN / TEST CSV
# ─────────────────────────────────────────────────────────────────

def build_native_dataset(
    raw_dir: str,
    train_csv_path: str,
    test_csv_path: str,
    train_per_lang: int = 50_000,
    test_per_lang:  int = 30_000,
):
    """
    For each of the 12+1 (English) languages:
      • Load train.txt → clean → take up to `train_per_lang` rows
      • Load test.txt  → clean → take up to `test_per_lang` rows
    Concatenate all languages into one CSV each.

    The final CSV columns are:
        Sentence   : cleaned text
        Language   : integer label (0 = English, 1-12 = Indian)
    """
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)

    train_dfs, test_dfs = [], []

    # LANGUAGE_CODES = {name: iso_code}.  We also add English.
    all_langs = dict(LANGUAGE_CODES)
    all_langs["English"] = "en"   # ensure English is included

    print("\n── Building NATIVE training set ──")
    for lang_name, lang_code in tqdm(all_langs.items(), desc="Train"):
        label = LABEL_TO_ID.get(lang_name, -1)
        if label == -1:
            print(f"  WARNING: No label found for {lang_name}, skipping.")
            continue
        fp = os.path.join(raw_dir, lang_code, "train.txt")
        df = load_language_file(fp, lang_name, label, train_per_lang)
        if not df.empty:
            train_dfs.append(df)

    print("\n── Building NATIVE test set ──")
    for lang_name, lang_code in tqdm(all_langs.items(), desc="Test"):
        label = LABEL_TO_ID.get(lang_name, -1)
        if label == -1:
            continue
        fp = os.path.join(raw_dir, lang_code, "test.txt")
        df = load_language_file(fp, lang_name, label, test_per_lang)
        if not df.empty:
            test_dfs.append(df)

    # Merge and shuffle
    train_df = pd.concat(train_dfs, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    test_df = pd.concat(test_dfs, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    # Drop rows with missing values
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Save
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
    test_df.to_csv(test_csv_path,   index=False, encoding="utf-8")

    print(f"\n  NATIVE TRAIN: {len(train_df):,} rows  →  {train_csv_path}")
    print(f"  NATIVE TEST : {len(test_df):,} rows  →  {test_csv_path}")

    # ── Class distribution report ──
    print("\n  Language distribution in TRAIN set:")
    dist = train_df["Language"].value_counts().sort_index()
    for label_id, count in dist.items():
        lang = LANGUAGE_LABELS.get(label_id, "Unknown")
        print(f"    [{label_id:2d}] {lang:12s}  →  {count:,}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Native Preprocessing   ║")
    print("╚══════════════════════════════════════════╝")
    print("\nThis script:")
    print("  1. Reads raw .txt files from data/raw/native/<lang_code>/")
    print("  2. Cleans each sentence (removes numbers, punctuation,")
    print("     and Latin chars for non-English languages)")
    print("  3. Saves to data/processed/native_train.csv & native_test.csv")
    print("  Each CSV has columns: Sentence, Language (integer label)\n")

    build_native_dataset(
        raw_dir=RAW_NATIVE_DIR,
        train_csv_path=NATIVE_TRAIN_CSV,
        test_csv_path=NATIVE_TEST_CSV,
        train_per_lang=50_000,
        test_per_lang=30_000,
    )

    print("\n✅ Native preprocessing complete.")
    print("Next step: Run  python 03_preprocess_romanized.py")
