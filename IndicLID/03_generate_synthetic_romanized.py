# ==============================================================
# 03_generate_synthetic_romanized.py
#
# STEP 3 — Generate Synthetic Romanized Training Data
#
# Paper methodology (Section 3.1):
#   Native training sentences → IndicXlit (Indic-to-English) → Romanized
#
# Our implementation:
#   The paper uses AI4Bharat IndicXlit (neural).
#   On Windows, IndicXlit's fairseq dependency fails to install due
#   to a symlink privilege error ([WinError 1314]).
#   We use `indic-transliteration` (pure-Python, rule-based, ITRANS scheme)
#   as a drop-in replacement.
#
#   Quality trade-off vs paper:
#   • Paper IndicXlit: ~96% acc on synthetic test set (Table 5)
#   • Our ITRANS:  linguistically accurate (ISO/IAST-style),
#                  each language produces distinct character distributions
#                  which is sufficient for language identification training.
#   • Perso-Arabic scripts (ur, ks, sd): handled via a custom
#     Unicode → Latin transliteration table (ALA-LC style).
#
# NOTE:
#   • Dogri (doi) is NOT supported by IndicXlit — paper acknowledges this.
#     We likewise skip doi from romanized training.
#   • Santali (sat) has no romanized class in IndicLID.
#
# Run:  python 03_generate_synthetic_romanized.py
# ==============================================================

import os
import re
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NATIVE_TRAIN_CSV,
    PROCESSED_DIR,
    SYNTHETIC_ROMANIZED_DIR,
    ROMANIZED_TRAIN_FT_TXT,
    ROMANIZED_TEST_FT_TXT,
    ROMANIZED_TRAIN_CSV,
    ROMANIZED_TEST_CSV,
    INDICXLIT_SUPPORTED,
    INDIC_LANGUAGES,
    ISO_TO_ROMAN_LABEL,
    ID_TO_LABEL,
)

# ─────────────────────────────────────────────────────────────────
# TRANSLITERATOR — indic-transliteration (ITRANS scheme)
# Pure-Python, no fairseq required.
# ─────────────────────────────────────────────────────────────────

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _it_translit

# Map ISO code → source script constant
_SCRIPT_MAP = {
    "hi":  sanscript.DEVANAGARI,
    "mr":  sanscript.DEVANAGARI,
    "ne":  sanscript.DEVANAGARI,
    "sa":  sanscript.DEVANAGARI,
    "kok": sanscript.DEVANAGARI,
    "mai": sanscript.DEVANAGARI,
    "doi": sanscript.DEVANAGARI,
    "brx": sanscript.DEVANAGARI,
    "bn":  sanscript.BENGALI,
    "as":  sanscript.BENGALI,
    "mni": sanscript.BENGALI,
    "gu":  sanscript.GUJARATI,
    "pa":  sanscript.GURMUKHI,
    "kn":  sanscript.KANNADA,
    "ml":  sanscript.MALAYALAM,
    "or":  sanscript.ORIYA,
    "ta":  sanscript.TAMIL,
    "te":  sanscript.TELUGU,
    # ur, ks, sd handled separately via Arabic transliteration table
}

# ─────────────────────────────────────────────────────────────────
# Perso-Arabic → Latin mapping (ALA-LC style, simplified)
# Covers standard Arabic + Urdu/Sindhi/Kashmiri extensions
# ─────────────────────────────────────────────────────────────────
_ARABIC_LATIN = {
    # Basic Arabic
    'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'T',
    'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh',
    'د': 'd', 'ڈ': 'D', 'ذ': 'z', 'ر': 'r', 'ڑ': 'R',
    'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh',
    'ف': 'f', 'ق': 'q', 'ک': 'k', 'ك': 'k', 'گ': 'g',
    'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'w',
    'ہ': 'h', 'ھ': 'h', 'ء': '', 'ی': 'y', 'ي': 'y',
    'ے': 'e', 'ئ': 'y',
    # Vowel marks (harakat)
    '\u064e': 'a', '\u064f': 'u', '\u0650': 'i',
    '\u064b': 'an', '\u064c': 'un', '\u064d': 'in',
    '\u0652': '',   # sukun
    '\u0651': '',   # shadda (gemination marker)
    '\u0670': 'a',  # alef above
    # Urdu-specific
    'آ': 'aa', 'اَ': 'a', 'اِ': 'i', 'اُ': 'u',
    'ٰ': 'a',
    # Digits
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}


def _arabic_to_latin(text: str) -> str:
    """Transliterate Perso-Arabic text to Latin using the mapping table."""
    result = []
    for ch in text:
        if ch in _ARABIC_LATIN:
            result.append(_ARABIC_LATIN[ch])
        elif ch == ' ':
            result.append(' ')
        elif ch.isascii():
            result.append(ch)
        # else: skip unmapped Perso-Arabic chars
    return ''.join(result)


def transliterate_sentence(sentence: str, iso_code: str) -> str:
    """
    Transliterate a native-script sentence to Roman/Latin.
    Uses indic-transliteration (IAST scheme) for Indic scripts — IAST
    produces word-level romanization with diacritics (\u0101, \u012b, \u1e6d …)
    which closely matches what IndicXlit (neural) would output.
    Custom ALA-LC mapping handles Perso-Arabic scripts (ur, ks, sd).
    """
    if not sentence or not sentence.strip():
        return sentence

    # Perso-Arabic languages
    if iso_code in ('ur', 'ks', 'sd'):
        roman = _arabic_to_latin(sentence)
    else:
        src_script = _SCRIPT_MAP.get(iso_code)
        if src_script is None:
            return sentence  # unsupported, return as-is
        try:
            roman = _it_translit(sentence, src_script, sanscript.IAST)
        except Exception:
            return sentence

    # Post-process: keep Latin letters (ASCII + IAST diacritics) and spaces
    # IAST diacritics live in Latin Extended-A/B (U+00C0-U+024F) and
    # Latin Extended Additional (U+1E00-U+1EFF: \u1e6d, \u1e63, \u1e41 …)
    roman = re.sub(r'[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s]', ' ', roman)
    roman = re.sub(r'\s+', ' ', roman).strip().lower()
    return roman


# ─────────────────────────────────────────────────────────────────
# ROMANIZED TEXT CLEANING
# ─────────────────────────────────────────────────────────────────

def clean_romanized(text: str) -> str:
    """
    Clean a romanized (Latin-script) string.

    Keeps ASCII letters plus IAST diacritics (\u0101=\u0101, \u012b=\u012b, \u1e6d=\u1e6d …)
    which are the output of indic-transliteration in IAST mode.
    These diacritics are language-distinctive and critical for LID training.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Remove digits
    text = re.sub(r'\d+', '', text)
    # 2. Keep Latin letters (including IAST extended-Latin diacritics) and spaces
    #    Strip anything else (Indic, Arabic, punctuation, symbols)
    text = re.sub(r'[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s]', ' ', text)
    # 3. Lowercase (Python .lower() handles Unicode correctly)
    text = text.lower()
    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_valid_romanized(text: str, min_chars: int = 3) -> bool:
    """At least min_chars non-space characters."""
    return len(text.replace(" ", "")) >= min_chars


# ─────────────────────────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────────────────────────

def generate_synthetic_romanized(
    native_csv: str,
    processed_dir: str,
    synthetic_dir: str,
    max_per_lang: int = 100_000,
    train_ratio: float = 0.85,
):
    """
    For each language in the native training CSV:
    1. Load native sentences
    2. Transliterate them to Roman script via IndicXlit
    3. Save per-language files + combined FastText / CSV files

    Parameters
    ----------
    native_csv     : path to native_train.csv
    processed_dir  : where to save combined files
    synthetic_dir  : where to save per-language .txt files
    max_per_lang   : max romanized sentences to generate per language
    train_ratio    : fraction to use for training (rest = test)
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    if not os.path.exists(native_csv):
        print(f"  ERROR: Native train CSV not found: {native_csv}")
        print("  Run 02_preprocess_native.py first.")
        return

    print("  Using indic-transliteration (ITRANS) for Indic scripts.")
    print("  Using ALA-LC table for Perso-Arabic scripts (ur, ks, sd).")

    df = pd.read_csv(native_csv, dtype=str).dropna()

    # Get unique iso codes present in the data
    all_isos = df["iso"].unique().tolist()

    all_train_ft = []
    all_test_ft  = []
    train_rows   = []
    test_rows    = []

    print(f"\n  Generating synthetic romanized data for {len(all_isos)} languages …")

    for iso_code in tqdm(all_isos, desc="  Languages"):
        lang_info = INDIC_LANGUAGES.get(iso_code)
        if lang_info is None:
            if iso_code == "en":
                # English sentences → label as "en", no transliteration needed
                # Just keep them as-is for the "Others/English" class
                continue  # skip English from romanized training
            continue

        lang_name, script, has_roman = lang_info
        if not has_roman:
            print(f"  [{iso_code}] {lang_name}: No romanized class. Skipping.")
            continue

        if iso_code not in INDICXLIT_SUPPORTED:
            print(f"  [{iso_code}] {lang_name}: IndicXlit not supported. Skipping.")
            # Paper note: doi (Dogri) is NOT supported by IndicXlit
            continue

        roman_label_id  = ISO_TO_ROMAN_LABEL.get(iso_code, -1)
        roman_label_str = ID_TO_LABEL.get(roman_label_id, f"{iso_code}_Latn")
        if roman_label_id < 0:
            continue

        # Filter sentences for this iso_code
        lang_df = df[df["iso"] == iso_code].copy()
        sentences = lang_df["sentence"].astype(str).tolist()

        if not sentences:
            print(f"  [{iso_code}] No sentences found in native CSV.")
            continue

        # Cap at max_per_lang
        sentences = sentences[:max_per_lang]

        print(f"  [{iso_code}] {lang_name:12s}: Transliterating {len(sentences):,} sentences …")

        # Transliterate using indic-transliteration
        romanized = []
        for sent in tqdm(sentences, desc=f"    [{iso_code}]", leave=False):
            romanized.append(transliterate_sentence(sent, iso_code))

        # Clean and filter
        clean_roman = []
        for r in romanized:
            c = clean_romanized(r)
            if is_valid_romanized(c):
                clean_roman.append(c)

        if not clean_roman:
            print(f"  [{iso_code}] No valid romanized sentences after cleaning.")
            continue

        # Oversample if too few
        if len(clean_roman) < max_per_lang // 4:
            print(f"  [{iso_code}] Low yield ({len(clean_roman):,}). Oversampling …")
            import random
            random.seed(42)
            while len(clean_roman) < max_per_lang // 4:
                clean_roman = clean_roman + clean_roman
            clean_roman = clean_roman[:max_per_lang]

        # Save per-language file
        lang_dir = os.path.join(synthetic_dir, iso_code)
        os.makedirs(lang_dir, exist_ok=True)
        lang_file = os.path.join(lang_dir, "train.txt")
        with open(lang_file, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_roman) + "\n")

        # Split into train / test
        n_train = int(len(clean_roman) * train_ratio)
        train_sents = clean_roman[:n_train]
        test_sents  = clean_roman[n_train:]

        print(f"  [{iso_code}] {lang_name:12s}: "
              f"romanized train={len(train_sents):,}  test={len(test_sents):,} "
              f"| label={roman_label_str}")

        for s in train_sents:
            all_train_ft.append(f"__label__{roman_label_str} {s}")
            train_rows.append({
                "sentence": s, "label": roman_label_id,
                "label_str": roman_label_str, "iso": iso_code
            })
        for s in test_sents:
            all_test_ft.append(f"__label__{roman_label_str} {s}")
            test_rows.append({
                "sentence": s, "label": roman_label_id,
                "label_str": roman_label_str, "iso": iso_code
            })

    # ── Shuffle ──
    import random
    random.seed(99)
    random.shuffle(all_train_ft)
    random.shuffle(all_test_ft)

    # ── Save FastText files ──
    print(f"\n  Saving romanized FastText train ({len(all_train_ft):,} lines) …")
    with open(ROMANIZED_TRAIN_FT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_train_ft) + "\n")
    print(f"  → {ROMANIZED_TRAIN_FT_TXT}")

    print(f"  Saving romanized FastText test ({len(all_test_ft):,} lines) …")
    with open(ROMANIZED_TEST_FT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_test_ft) + "\n")
    print(f"  → {ROMANIZED_TEST_FT_TXT}")

    # ── Save CSV ──
    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_rows)
    train_df.to_csv(ROMANIZED_TRAIN_CSV, index=False, encoding="utf-8")
    test_df.to_csv( ROMANIZED_TEST_CSV,  index=False, encoding="utf-8")
    print(f"  → {ROMANIZED_TRAIN_CSV}  ({len(train_df):,} rows)")
    print(f"  → {ROMANIZED_TEST_CSV}   ({len(test_df):,} rows)")

    # ── Distribution ──
    if not train_df.empty:
        print("\n  Class distribution in ROMANIZED train set:")
        dist = train_df.groupby("label_str").size().sort_values(ascending=False)
        for label_str, count in dist.items():
            print(f"    {label_str:20s} → {count:,}")

    return len(all_train_ft), len(all_test_ft)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  IndicLID — Step 3: Generate Synthetic Romanized Data   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\nThis script:")
    print("  1. Loads native-script sentences from native_train.csv")
    print("  2. Transliterates each sentence using AI4Bharat IndicXlit")
    print("     (Indic-to-English direction — the Romanized output)")
    print("  3. Cleans and validates the romanized output")
    print("  4. Saves per-language files + combined FastText/CSV files")
    print("\n  NOTE: This is Paper Section 3.1's key contribution —")
    print("  synthetic romanized data created via transliteration.")
    print("  Without this, there is no training data for romanized LID.\n")

    from config import NATIVE_TRAIN_CSV, PROCESSED_DIR, SYNTHETIC_ROMANIZED_DIR

    n_train, n_test = generate_synthetic_romanized(
        native_csv=NATIVE_TRAIN_CSV,
        processed_dir=PROCESSED_DIR,
        synthetic_dir=SYNTHETIC_ROMANIZED_DIR,
        max_per_lang=100_000,
        train_ratio=0.85,
    )

    print(f"\n{'='*60}")
    print(f"  Synthetic romanized generation complete.")
    print(f"  Total train lines: {n_train:,}")
    print(f"  Total test lines : {n_test:,}")
    print(f"  Next step: python 04_train_fasttext.py")
    print(f"{'='*60}")
