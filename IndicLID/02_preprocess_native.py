# ==============================================================
# 02_preprocess_native.py
#
# STEP 2 — Preprocess native-script sentences for IndicLID-FTN.
#
# Paper methodology (Section 3.1):
#   • Compile 100k sentences per language from IndicCorpV2
#   • Multiple sources: IndicCorp, NLLB, Wikipedia, Vikaspedia
#   • Tokenize and normalize using IndicNLP library
#   • Oversample low-resource languages to reach 100k
#
# Output:
#   • data/processed/native_train_ft.txt  ← FastText supervised format
#   • data/processed/native_test_ft.txt
#   • data/processed/native_train.csv     ← CSV for IndicBERT
#   • data/processed/native_test.csv
#
# FastText format: __label__<lang_code> <sentence>
# Example:         __label__hi_Deva नमस्ते आप कैसे हैं
#
# Run:  python 02_preprocess_native.py
# ==============================================================

import os
import re
import sys
import unicodedata
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# regex module supports \p{L}, \p{M} etc. (Unicode property classes)
# Required to NOT strip Indic combining marks (matras, halant, anusvara ...)
# Python's built-in re treats \w as letters only — matras like U+094B (ো)
# are category Mc (Spacing_Combining_Mark) and would be stripped by [^\w\s].
import regex as _rx

# IndicNLP normalization (paper: "tokenized and normalized using IndicNLP library
# with default settings" — Section 3.1)
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory as _IndicNormFactory
    _INDIC_NORM_FACTORY = _IndicNormFactory()
    _INDICNLP_AVAILABLE = True
except ImportError:
    _INDICNLP_AVAILABLE = False
    print("[WARN] indic-nlp-library not found — skipping IndicNLP normalization")

_NORMALIZER_CACHE: dict = {}


def _normalize_with_indicnlp(text: str, iso_code: str) -> str:
    """
    Apply IndicNLP normalization using default settings (as in paper Section 3.1).
    Falls back silently if language not supported or library unavailable.
    """
    if not _INDICNLP_AVAILABLE or iso_code == "en":
        return text
    if iso_code not in _NORMALIZER_CACHE:
        try:
            _NORMALIZER_CACHE[iso_code] = _INDIC_NORM_FACTORY.get_normalizer(iso_code)
        except Exception:
            _NORMALIZER_CACHE[iso_code] = None
    normalizer = _NORMALIZER_CACHE[iso_code]
    if normalizer is None:
        return text
    try:
        return normalizer.normalize(text)
    except Exception:
        return text

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_NATIVE_DIR,
    PROCESSED_DIR,
    INDIC_LANGUAGES,
    ISO_TO_NATIVE_LABEL,
    ID_TO_LABEL,
    TRAINING_SAMPLES_PER_LANG,
)

# ─────────────────────────────────────────────────────────────────
# UNICODE SCRIPT RANGES FOR VALIDATION
# Used to filter sentences that belong to the expected script
# ─────────────────────────────────────────────────────────────────
SCRIPT_RANGES = {
    "Bengali":      (0x0980, 0x09FF),
    "Devanagari":   (0x0900, 0x097F),
    "Gujarati":     (0x0A80, 0x0AFF),
    "Gurmukhi":     (0x0A00, 0x0A7F),
    "Kannada":      (0x0C80, 0x0CFF),
    "Malayalam":    (0x0D00, 0x0D7F),
    "Oriya":        (0x0B00, 0x0B7F),
    "Tamil":        (0x0B80, 0x0BFF),
    "Telugu":       (0x0C00, 0x0C7F),
    "Perso-Arabic": (0x0600, 0x06FF),
    "Ol Chiki":     (0x1C50, 0x1C7F),
    "Meetei Mayek": (0xABC0, 0xABFF),
}

# Languages for which we skip script validation (mixed/special)
SKIP_SCRIPT_VALIDATION = {"doi", "kok", "mai", "ne", "brx", "sa"}


def get_script_fraction(text: str, script_name: str) -> float:
    """
    Return fraction of characters that belong to the expected script.
    Used to validate that a sentence is actually in the correct script.
    """
    if script_name not in SCRIPT_RANGES:
        return 1.0   # skip validation for unknown scripts

    lo, hi = SCRIPT_RANGES[script_name]
    total   = 0
    in_script = 0

    for ch in text:
        if ch.isalpha():
            total += 1
            cp = ord(ch)
            if lo <= cp <= hi:
                in_script += 1

    return (in_script / total) if total > 0 else 0.0


def clean_native_sentence(text: str, iso_code: str, script: str) -> str:
    """
    Clean a native-script sentence following the paper (Section 3.1).

    Steps:
    1. Unicode normalize (NFC)
    2. Remove digits (paper: remove numbers)
    3. Remove punctuation and special symbols
    4. For non-Perso-Arabic scripts: remove Latin characters
       (prevents cross-script contamination)
    5. Collapse multiple spaces → single space
    6. Strip
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 0. IndicNLP normalization (paper: "IndicNLP library with default settings")
    text = _normalize_with_indicnlp(text, iso_code)

    # 1. Unicode normalize
    text = unicodedata.normalize("NFC", text)

    # 2. Remove all Unicode digits (\p{N} catches Devanagari, Arabic-Indic,
    #    Bengali, Tamil numerals etc. — re's '\d' only catches ASCII/some)
    text = _rx.sub(r'\p{N}+', '', text)

    # 3. Remove punctuation/symbols but PRESERVE all Unicode letters (\p{L})
    #    AND combining marks (\p{M}: matras, halant, anusvara, chandrabindu …)
    #    Python's re [^\w\s] would strip Mc/Mn combining chars — don't use it.
    text = _rx.sub(r'[^\p{L}\p{M}\s]', ' ', text)

    # 4. Remove Latin/ASCII letters for non-English native scripts
    #    (Latin contamination would confuse script detection)
    if iso_code != "en":
        text = re.sub(r'[a-zA-Z]', '', text)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_valid_native(text: str, iso_code: str, script: str,
                    min_chars: int = 5) -> bool:
    """
    Validate a cleaned native-script sentence:
    1. Minimum character length
    2. Majority of alphabetic chars must be in the correct script
    """
    if len(text) < min_chars:
        return False

    if iso_code in SKIP_SCRIPT_VALIDATION:
        # These use Devanagari but overlap with Hindi — trust the source
        return True

    frac = get_script_fraction(text, script)
    return frac >= 0.5   # at least half must be in correct script


# ─────────────────────────────────────────────────────────────────
# LOAD ONE LANGUAGE FILE
# ─────────────────────────────────────────────────────────────────

def load_language(
    raw_dir: str,
    iso_code: str,
    lang_name: str,
    script: str,
    n_max: int,
) -> list:
    """
    Load and clean up to n_max sentences for one language.
    Returns list of cleaned sentence strings.
    """
    filepath = os.path.join(raw_dir, iso_code, "train.txt")

    if not os.path.exists(filepath):
        print(f"    [{iso_code}] File not found: {filepath}")
        return []

    rows = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(rows) >= n_max:
                break
            cleaned = clean_native_sentence(line.strip(), iso_code, script)
            if is_valid_native(cleaned, iso_code, script):
                rows.append(cleaned)

    return rows


# ─────────────────────────────────────────────────────────────────
# BUILD DATASETS
# ─────────────────────────────────────────────────────────────────
# Train/validation split ratio.
# The paper trains on ALL 100k sentences per language and tests on the
# Bhasha-Abhijnaanam benchmark (separate held-out set).
# We keep 5% as a small internal validation set for IndicBERT training
# monitoring; FastText doesn't need it but we generate it for consistency.
TRAIN_RATIO = 0.95


def build_native_datasets(
    raw_dir: str,
    processed_dir: str,
    n_per_lang: int = TRAINING_SAMPLES_PER_LANG,
):
    """
    Build the native-script FastText training files and CSVs.
    """
    os.makedirs(processed_dir, exist_ok=True)

    all_train_ft = []   # list of FastText-format strings
    all_test_ft  = []
    train_rows   = []   # list of (sentence, label_id, label_str, iso)
    test_rows    = []

    print("\n── Loading and cleaning native-script data ──")

    # Add English as well
    langs_to_process = dict(INDIC_LANGUAGES)
    # English is in INDICCORPV2 but not in INDIC_LANGUAGES; add it
    langs_english = {"en": ("English", "Latin", True)}

    for iso_code, (lang_name, script, _) in tqdm(
        {**langs_to_process, **langs_english}.items(), desc="Languages"
    ):
        label_str = _get_native_label_str(iso_code)
        label_id  = ISO_TO_NATIVE_LABEL.get(iso_code, -1)

        if label_id < 0:
            continue  # not in our label map

        sentences = load_language(raw_dir, iso_code, lang_name, script, n_per_lang)

        if not sentences:
            print(f"  [{iso_code}] No valid sentences found, skipping.")
            continue

        # Oversample if below target
        if len(sentences) < n_per_lang // 2:
            # Duplicate data to reach at least half the target
            while len(sentences) < n_per_lang // 2:
                sentences = sentences + sentences
            sentences = sentences[:n_per_lang]
            print(f"  [{iso_code}] Low-resource: oversampled to {len(sentences):,}")

        # Train/test split
        n_train = int(len(sentences) * TRAIN_RATIO)
        train_sents = sentences[:n_train]
        test_sents  = sentences[n_train:]

        print(f"  [{iso_code}] {lang_name:12s} | train={len(train_sents):,} "
              f"test={len(test_sents):,} | label={label_str}")

        # FastText format: __label__<label> sentence
        for s in train_sents:
            all_train_ft.append(f"__label__{label_str} {s}")
            train_rows.append({
                "sentence": s, "label": label_id,
                "label_str": label_str, "iso": iso_code
            })
        for s in test_sents:
            all_test_ft.append(f"__label__{label_str} {s}")
            test_rows.append({
                "sentence": s, "label": label_id,
                "label_str": label_str, "iso": iso_code
            })

    # ── Shuffle ──
    import random
    random.seed(42)
    random.shuffle(all_train_ft)
    random.shuffle(all_test_ft)

    # ── Save FastText format ──
    from config import NATIVE_TRAIN_FT_TXT, NATIVE_TEST_FT_TXT
    from config import NATIVE_TRAIN_CSV, NATIVE_TEST_CSV

    print(f"\n  Saving FastText train file ({len(all_train_ft):,} lines) …")
    with open(NATIVE_TRAIN_FT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_train_ft) + "\n")
    print(f"  → {NATIVE_TRAIN_FT_TXT}")

    print(f"  Saving FastText test file ({len(all_test_ft):,} lines) …")
    with open(NATIVE_TEST_FT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_test_ft) + "\n")
    print(f"  → {NATIVE_TEST_FT_TXT}")

    # ── Save CSV ──
    train_df = pd.DataFrame(train_rows).dropna()
    test_df  = pd.DataFrame(test_rows).dropna()

    train_df.to_csv(NATIVE_TRAIN_CSV, index=False, encoding="utf-8")
    test_df.to_csv( NATIVE_TEST_CSV,  index=False, encoding="utf-8")
    print(f"  → {NATIVE_TRAIN_CSV}  ({len(train_df):,} rows)")
    print(f"  → {NATIVE_TEST_CSV}   ({len(test_df):,} rows)")

    # ── Class distribution ──
    print("\n  Class distribution in NATIVE train set:")
    dist = train_df.groupby("label_str").size().sort_values(ascending=False)
    for label_str, count in dist.items():
        print(f"    {label_str:20s} → {count:,}")

    return len(all_train_ft), len(all_test_ft)


def _get_native_label_str(iso_code: str) -> str:
    """
    Return the label string for FastText training format.
    E.g. 'hi' → 'hi_Deva', 'bn' → 'bn_Beng', 'en' → 'en'
    """
    # Map iso → label ID → label string
    label_id = ISO_TO_NATIVE_LABEL.get(iso_code, -1)
    if label_id >= 0:
        return ID_TO_LABEL.get(label_id, iso_code)
    return iso_code


# ─────────────────────────────────────────────────────────────────
# ALSO: Prepare existing raw/ data for Kashmiri (dual script)
# Kashmiri appears in BOTH Perso-Arabic (primary) and Devanagari.
# The paper treats these as 2 separate native-script classes.
# ─────────────────────────────────────────────────────────────────

def prepare_kashmiri_devanagari(raw_dir: str, processed_dir: str):
    """
    If Kashmiri Devanagari (ks_Deva, label=8) data is available,
    add it to both the FastText file and CSV.
    """
    # IndicCorpV2 has Kashmiri in Perso-Arabic (ks).
    # Devanagari Kashmiri is rare and may not be in IndicCorpV2.
    # We note it here but skip gracefully if no data.
    ks_deva_dir = os.path.join(raw_dir, "ks_Deva")
    if not os.path.isdir(ks_deva_dir):
        return  # Not available, skip

    from config import NATIVE_TRAIN_FT_TXT
    fp = os.path.join(ks_deva_dir, "train.txt")
    if not os.path.exists(fp):
        return

    print("  Adding Kashmiri (Devanagari) sentences …")
    rows = []
    with open(fp, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = clean_native_sentence(line.strip(), "ks", "Devanagari")
            if is_valid_native(s, "ks", "Devanagari"):
                rows.append(s)

    if rows:
        with open(NATIVE_TRAIN_FT_TXT, "a", encoding="utf-8") as f:
            for s in rows:
                f.write(f"__label__ks_Deva {s}\n")
        print(f"  Added {len(rows):,} Kashmiri (Devanagari) sentences.")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  IndicLID — Step 2: Preprocess Native Script Data   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\nThis script:")
    print("  1. Reads raw native .txt files from data/raw/native/")
    print("  2. Cleans sentences (removes numbers, punctuation,")
    print("     Latin chars for non-English scripts)")
    print("  3. Validates each sentence belongs to the expected script")
    print("  4. Saves FastText format (.txt) and CSV files")
    print(f"  5. Target: {TRAINING_SAMPLES_PER_LANG:,} sentences per language\n")

    n_train, n_test = build_native_datasets(
        raw_dir=RAW_NATIVE_DIR,
        processed_dir=PROCESSED_DIR,
        n_per_lang=TRAINING_SAMPLES_PER_LANG,
    )

    prepare_kashmiri_devanagari(RAW_NATIVE_DIR, PROCESSED_DIR)

    print(f"\n{'='*60}")
    print(f"  Native preprocessing complete.")
    print(f"  Total train lines: {n_train:,}")
    print(f"  Total test lines : {n_test:,}")
    print(f"  Next step: python 03_generate_synthetic_romanized.py")
    print(f"{'='*60}")
