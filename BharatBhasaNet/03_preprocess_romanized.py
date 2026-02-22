# ==============================================================
# 03_preprocess_romanized.py
#
# STEP 3 — Preprocess Romanized sentences.
#
# What the paper did (Section III-B):
#   • Sources: Aksharantar (train/test) + Bhasha-Abhijnaanam (LID)
#   • Remove special characters, numbers, and punctuation
#   • Combine all sources into a single train CSV
#   • For testing, use ONLY Aksharantar (Valid) split
#   • Output CSV columns: "Sentence", "Language" (int label)
#
# Run:  python 03_preprocess_romanized.py
# ==============================================================

import os
import re
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_ROMANIZED_DIR,
    PROCESSED_DIR,
    LANGUAGE_LABELS,
    LANGUAGE_CODES,
    LABEL_TO_ID,
    ROMANIZED_TRAIN_CSV,
    ROMANIZED_TEST_CSV,
)


# ─────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────

def clean_romanized(text: str) -> str:
    """
    Clean a romanized (Latin-script) sentence.

    Romanized sentences are KEPT in Latin script — we must NOT
    remove Latin characters here (unlike native preprocessing).

    Steps:
    1. Remove digits
    2. Remove non-alphanumeric characters (keep spaces)
    3. Lower-case (optional but helps consistency)
    4. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove digits
    text = re.sub(r'\d+', '', text)

    # 2. Keep only letters (Latin) and spaces
    #    Allow Unicode letters so Devanagari/other scripts in
    #    mixed files are handled gracefully (they'll be filtered later)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('_', ' ')

    # 3. Keep only Latin letters and spaces for purely romanized data
    #    (strips any accidentally included native-script characters)
    text = re.sub(r'[^\x00-\x7F\s]', '', text)

    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text


def is_valid(text: str, min_chars: int = 3) -> bool:
    return len(text) >= min_chars


# ─────────────────────────────────────────────────────────────────
# LOAD FROM RAW TEXT FILES
# ─────────────────────────────────────────────────────────────────

def load_romanized_file(
    filepath: str,
    lang_name: str,
    lang_label: int,
    n_max: int = 999_999,
) -> pd.DataFrame:
    """Load and clean a single romanized .txt file."""
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=["Sentence", "Language"])

    rows = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(rows) >= n_max:
                break
            cleaned = clean_romanized(line)
            if is_valid(cleaned):
                rows.append({"Sentence": cleaned, "Language": lang_label})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# SOURCE 1 — AKSHARANTAR
#   Folder structure after download:
#     data/raw/romanized/<lang_code>/train.txt
#     data/raw/romanized/<lang_code>/test.txt   (used as TEST set)
#     data/raw/romanized/<lang_code>/valid.txt
# ─────────────────────────────────────────────────────────────────

def load_aksharantar(raw_dir: str) -> tuple:
    """
    Returns (train_df, test_df) loaded from Aksharantar files.
    Test set = 'valid' split (as stated in the paper).
    """
    AKSHARANTAR_LANGS = {
        "Bengali":   "bn",
        "Assamese":  "as",
        "Hindi":     "hi",
        "Marathi":   "mr",
        "Tamil":     "ta",
        "Telugu":    "te",
        "Kannada":   "kn",
        "Malayalam": "ml",
        "Gujarati":  "gu",
        "Oriya":     "or",
        "Punjabi":   "pa",
        "Urdu":      "ur",
    }

    train_dfs, test_dfs = [], []

    print("\n  Loading Aksharantar …")
    for lang_name, lang_code in tqdm(AKSHARANTAR_LANGS.items(), desc="Aksharantar"):
        label = LABEL_TO_ID.get(lang_name, -1)
        if label == -1:
            continue

        lang_dir = os.path.join(raw_dir, lang_code)

        # TRAIN: combine train.txt and test.txt from Aksharantar
        for split_name in ["train", "test"]:
            fp = os.path.join(lang_dir, f"{split_name}.txt")
            df = load_romanized_file(fp, lang_name, label)
            if not df.empty:
                train_dfs.append(df)
                print(f"    {lang_name:10s} [{split_name}] → {len(df):,} rows")

        # TEST: use 'valid' split only (paper Section III-B)
        fp_valid = os.path.join(lang_dir, "valid.txt")
        df_test  = load_romanized_file(fp_valid, lang_name, label)
        if not df_test.empty:
            test_dfs.append(df_test)
            print(f"    {lang_name:10s} [valid]  → {len(df_test):,} rows (TEST)")

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df  = pd.concat(test_dfs,  ignore_index=True) if test_dfs  else pd.DataFrame()

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────
# SOURCE 2 — BHASHA-ABHIJNAANAM
#   Folder structure after download:
#     data/raw/romanized/bhasha_abhijnaanam/<lang_label>/train.txt
# ─────────────────────────────────────────────────────────────────

def load_bhasha_abhijnaanam(raw_dir: str) -> pd.DataFrame:
    """
    Load the Bhasha-Abhijnaanam LID dataset (romanized only).
    Returns a DataFrame with columns ['Sentence', 'Language'].
    """
    bhasha_dir = os.path.join(raw_dir, "bhasha_abhijnaanam")
    if not os.path.isdir(bhasha_dir):
        print(f"  Bhasha-Abhijnaanam folder not found: {bhasha_dir}")
        return pd.DataFrame()

    print("\n  Loading Bhasha-Abhijnaanam …")

    # Map the folder names (which are language codes or names) to int labels
    all_dfs = []
    for folder in os.listdir(bhasha_dir):
        folder_path = os.path.join(bhasha_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Try to map folder name to label
        label = LABEL_TO_ID.get(folder) or LABEL_TO_ID.get(
            {v: k for k, v in LANGUAGE_CODES.items()}.get(folder, ""), -1
        )
        if label == -1:
            # Try direct integer conversion (folder name might be the label)
            try:
                label = int(folder)
            except ValueError:
                print(f"    Skipping unknown folder: {folder}")
                continue

        lang_name = LANGUAGE_LABELS.get(label, f"lang_{label}")

        for split_file in ["train.txt", "valid.txt", "test.txt"]:
            fp = os.path.join(folder_path, split_file)
            df = load_romanized_file(fp, lang_name, label)
            if not df.empty:
                all_dfs.append(df)
                print(f"    {lang_name:10s} [{split_file}] → {len(df):,} rows")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# BUILD FULL ROMANIZED TRAIN / TEST CSV
# ─────────────────────────────────────────────────────────────────

def build_romanized_dataset(
    raw_dir: str,
    train_csv_path: str,
    test_csv_path:  str,
):
    """
    Merge all romanized sources → single train.csv and test.csv.
    """
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)

    # ── Load Aksharantar ──
    aksharantar_train, aksharantar_test = load_aksharantar(raw_dir)

    # ── Load Bhasha-Abhijnaanam (training only) ──
    bhasha_train = load_bhasha_abhijnaanam(raw_dir)

    # ── Combine all training sources ──
    train_parts = [df for df in [aksharantar_train, bhasha_train] if not df.empty]
    if not train_parts:
        print("  ERROR: No romanized training data found. Run 01_download_datasets.py first.")
        return

    train_df = pd.concat(train_parts, ignore_index=True)

    # ── Test = Aksharantar valid split only ──
    test_df = aksharantar_test

    # ── Clean up ──
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # ── Shuffle ──
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Save ──
    train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
    test_df.to_csv( test_csv_path,  index=False, encoding="utf-8")

    print(f"\n  ROMANIZED TRAIN: {len(train_df):,} rows  →  {train_csv_path}")
    print(f"  ROMANIZED TEST : {len(test_df):,} rows  →  {test_csv_path}")

    print("\n  Language distribution in ROMANIZED TRAIN:")
    dist = train_df["Language"].value_counts().sort_index()
    for label_id, count in dist.items():
        lang = LANGUAGE_LABELS.get(label_id, "Unknown")
        print(f"    [{label_id:2d}] {lang:12s}  →  {count:,}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Romanized Preprocessing     ║")
    print("╚════════════════════════════════════════════════╝")
    print("\nThis script:")
    print("  1. Reads raw romanized .txt files from data/raw/romanized/")
    print("  2. Cleans each sentence (removes digits, punctuation)")
    print("  3. Combines Aksharantar + Bhasha-Abhijnaanam for training")
    print("  4. Uses only Aksharantar (valid) as the test set")
    print("  5. Saves to data/processed/romanized_train.csv & romanized_test.csv\n")

    build_romanized_dataset(
        raw_dir=RAW_ROMANIZED_DIR,
        train_csv_path=ROMANIZED_TRAIN_CSV,
        test_csv_path=ROMANIZED_TEST_CSV,
    )

    print("\n✅ Romanized preprocessing complete.")
    print("Next step: Run  python 04_train_native.py")
