# ==============================================================
# 01_download_datasets.py
#
# STEP 1 — Download all datasets used in the paper:
#   1. IndicCorpV2   → Native script sentences (HTTP streaming)
#   2. Aksharantar   → Romanized (transliteration) word pairs
#   3. Bhasha-Abhijnaanam → Romanized language identification data
#
# Run:  python 01_download_datasets.py
# ==============================================================

import io
import os
import sys
import zipfile
import requests
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW_NATIVE_DIR,
    RAW_ROMANIZED_DIR,
    LANGUAGE_CODES,
)
from datasets import load_dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────
def save_text_to_file(texts: list, filepath: str):
    """Save a list of strings, one per line, to a .txt file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")
    print(f"    Saved {len(texts):,} lines → {filepath}")


def stream_indiccorpv2_lines(lang_code: str, n_lines: int) -> list:
    """
    Stream the first n_lines non-empty lines from IndicCorpV2
    for the given language code by reading the raw HuggingFace URL.
    This avoids downloading the full multi-GB file.
    """
    base_url = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data"
    url = f"{base_url}/{lang_code}.txt"

    lines = []
    print(f"    Streaming: {url}")

    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                text = raw.strip()
                if text:
                    lines.append(text)
                if len(lines) >= n_lines:
                    break
    except requests.exceptions.RequestException as e:
        print(f"    ERROR: {e}")

    return lines


# ─────────────────────────────────────────────────────────────────
# DATASET 1 — IndicCorpV2 (Native Scripts + English)
# Files: ai4bharat/IndicCorpV2/resolve/main/data/<lang_code>.txt
# Each file is plain UTF-8, one sentence per line.
# We only need 50k train + 30k test, so we stream the first 80k
# lines per language (avoiding download of full multi-GB files).
# ─────────────────────────────────────────────────────────────────
def download_indiccorpv2(
    save_dir:       str,
    train_per_lang: int = 50_000,
    test_per_lang:  int = 30_000,
):
    print("\n" + "=" * 60)
    print("DOWNLOADING IndicCorpV2 (Native script + English)")
    print("=" * 60)

    # Language name → file code in IndicCorpV2 data/ directory
    # Verified from: https://huggingface.co/datasets/ai4bharat/IndicCorpV2/tree/main/data
    INDICCORPV2_CODES = {
        "Bengali":   "bn",
        "Assamese":  "as",
        "Hindi":     "hi-1",   # Hindi split into hi-1.txt / hi-2.txt / hi-3.txt
        "Marathi":   "mr",
        "Tamil":     "ta",
        "Telugu":    "te",
        "Kannada":   "kn",
        "Malayalam": "ml",
        "Gujarati":  "gu",
        "Oriya":     "or",
        "Urdu":      "ur",
        "Punjabi":   "pa",
        "English":   "en",
    }

    total_needed = train_per_lang + test_per_lang

    for lang_name, file_code in tqdm(INDICCORPV2_CODES.items(), desc="Languages"):
        # Determine the 2-letter save directory code
        if lang_name == "English":
            save_code = "en"
        else:
            save_code = LANGUAGE_CODES.get(lang_name, file_code)

        out_dir    = os.path.join(save_dir, save_code)
        train_path = os.path.join(out_dir, "train.txt")
        test_path  = os.path.join(out_dir, "test.txt")

        print(f"\n  [{lang_name}]  file={file_code}.txt  save_dir={save_code}/")

        # Skip if already downloaded
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"    Already exists — skipping.")
            continue

        lines = stream_indiccorpv2_lines(file_code, total_needed)

        if not lines:
            print(f"    WARNING: No lines fetched for {lang_name}.")
            continue

        print(f"    Fetched {len(lines):,} lines.")
        save_text_to_file(lines[:train_per_lang], train_path)
        save_text_to_file(lines[train_per_lang:], test_path)

    print("\n  IndicCorpV2 download complete.")


# ─────────────────────────────────────────────────────────────────
# DATASET 2 — Aksharantar (Romanized / Transliteration)
# HuggingFace: ai4bharat/Aksharantar
# The dataset has per-language ZIP files in the repo root:
#   https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/{lang3}.zip
# Each ZIP contains train/test/valid TSV files with columns:
#   unique_identifier, native word, english word, source, score
# We save only the 'english word' (romanized) column per language.
# ─────────────────────────────────────────────────────────────────
def download_aksharantar(save_dir: str):
    print("\n" + "=" * 60)
    print("DOWNLOADING Aksharantar (Romanized word pairs, direct ZIP)")
    print("=" * 60)

    # Maps our 2-letter save code → 3-letter Aksharantar ZIP filename code
    AKSHARANTAR_LANGS = {
        "bn": "ben",   # Bengali
        "as": "asm",   # Assamese
        "hi": "hin",   # Hindi
        "mr": "mar",   # Marathi
        "ta": "tam",   # Tamil
        "te": "tel",   # Telugu
        "kn": "kan",   # Kannada
        "ml": "mal",   # Malayalam
        "gu": "guj",   # Gujarati
        "or": "ori",   # Oriya
        "pa": "pan",   # Punjabi
        "ur": "urd",   # Urdu
    }

    BASE_URL = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main"

    for save_code, zip_code in tqdm(AKSHARANTAR_LANGS.items(), desc="Languages"):
        lang_dir = os.path.join(save_dir, save_code)

        already_done = all(
            os.path.exists(os.path.join(lang_dir, f"{s}.txt"))
            for s in ["train", "test", "valid"]
        )
        if already_done:
            print(f"  [{zip_code}] already exists — skipping.")
            continue

        url = f"{BASE_URL}/{zip_code}.zip"
        print(f"\n  [{zip_code}] Downloading {url} …")

        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"    ERROR downloading ZIP: {e}")
            continue

        print(f"    Downloaded {len(resp.content):,} bytes. Extracting …")

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                # List all files in the ZIP
                all_files = zf.namelist()
                print(f"    ZIP contents: {all_files}")

                for fname in all_files:
                    # Determine split from filename
                    low = fname.lower()
                    if "train" in low:
                        split_out = "train"
                    elif "test" in low:
                        split_out = "test"
                    elif "valid" in low or "val" in low or "dev" in low:
                        split_out = "valid"
                    else:
                        continue  # skip README etc.

                    out_path = os.path.join(lang_dir, f"{split_out}.txt")
                    if os.path.exists(out_path):
                        continue

                    with zf.open(fname) as f:
                        content = f.read().decode("utf-8", errors="replace")

                    lines = content.splitlines()
                    if not lines:
                        continue

                    # Parse header to find the romanized column
                    # Typical TSV header: unique_identifier\tnative word\tenglish word\tsource\tscore
                    header = lines[0].split("\t")
                    rom_idx = None
                    for i, h in enumerate(header):
                        if "english" in h.lower() or "roman" in h.lower():
                            rom_idx = i
                            break

                    if rom_idx is None:
                        # Try comma-separated
                        header = lines[0].split(",")
                        for i, h in enumerate(header):
                            if "english" in h.lower() or "roman" in h.lower():
                                rom_idx = i
                                break
                        sep = ","
                    else:
                        sep = "\t"

                    if rom_idx is None:
                        print(f"    Could not find romanized column in: {header}")
                        continue

                    words = []
                    for line in lines[1:]:
                        parts = line.split(sep)
                        if len(parts) > rom_idx:
                            w = parts[rom_idx].strip()
                            if w:
                                words.append(w)

                    save_text_to_file(words, out_path)

        except zipfile.BadZipFile as e:
            print(f"    ERROR: Bad ZIP file: {e}")
            continue

    print("\n  Aksharantar download complete.")


# ─────────────────────────────────────────────────────────────────
# DATASET 3 — Bhasha-Abhijnaanam (Romanized LID)
# HuggingFace: ai4bharat/Bhasha-Abhijnaanam
# Single ZIP file: bhasha-abhijnaanam.zip (10.7 MB)
# Contains labeled sentences in both native and romanized script.
# Used as the test set for the romanized model in the paper.
#
# Expected JSON keys: unique_identifier, native sentence,
#                     romanized sentence, language, script, source
# ─────────────────────────────────────────────────────────────────
def download_bhasha_abhijnaanam(save_dir: str):
    print("\n" + "=" * 60)
    print("DOWNLOADING Bhasha-Abhijnaanam (Romanized LID test set)")
    print("=" * 60)

    # Dataset language name → our 2-letter codes
    LANG_NAME_TO_CODE = {
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
        "Urdu":      "ur",
        "Punjabi":   "pa",
        "English":   "en",
    }

    url = "https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam/resolve/main/bhasha-abhijnaanam.zip"
    print(f"  Downloading {url} …")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Downloaded {len(resp.content):,} bytes. Extracting …")

    import json

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            all_files = zf.namelist()
            print(f"  ZIP contents ({len(all_files)} files): {all_files[:10]} …")

            # Collect all romanized sentences grouped by language
            lang_rom_sentences: dict = {}   # code → list[str]
            lang_nat_sentences: dict = {}   # code → list[str]

            for fname in all_files:
                low = fname.lower()
                if not (low.endswith(".json") or low.endswith(".tsv") or low.endswith(".csv")):
                    continue

                with zf.open(fname) as f:
                    raw = f.read().decode("utf-8", errors="replace")

                if low.endswith(".json"):
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        # JSONL
                        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

                    # Normalise to a flat list of {lang, romanized, native} dicts
                    records = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                records.append(item)
                    elif isinstance(data, dict):
                        first_val = next(iter(data.values()), None)
                        if isinstance(first_val, list):
                            # {"lang_code": [record, ...], ...}
                            for lang_key, items in data.items():
                                for item in items:
                                    if isinstance(item, dict):
                                        if "language" not in item:
                                            item = {**item, "language": lang_key}
                                        records.append(item)
                        elif isinstance(first_val, dict):
                            # {"id": {record}, ...}
                            records = list(data.values())
                        else:
                            records = [data]

                    print(f"    {fname}: {len(records)} records total")

                    for item in records:
                        lang_name = (item.get("language") or "").strip()
                        rom       = (item.get("romanized sentence") or
                                     item.get("romanized_sentence") or "").strip()
                        nat       = (item.get("native sentence") or
                                     item.get("native_sentence") or "").strip()
                        code = LANG_NAME_TO_CODE.get(lang_name)
                        if code and rom:
                            lang_rom_sentences.setdefault(code, []).append(rom)
                        if code and nat:
                            lang_nat_sentences.setdefault(code, []).append(nat)

                elif low.endswith(".tsv") or low.endswith(".csv"):
                    sep   = "\t" if low.endswith(".tsv") else ","
                    lines = raw.splitlines()
                    if not lines:
                        continue
                    header = lines[0].split(sep)
                    rom_i,  lang_i, nat_i = None, None, None
                    for i, h in enumerate(header):
                        h_low = h.lower().strip()
                        if "roman" in h_low:
                            rom_i = i
                        elif "language" in h_low:
                            lang_i = i
                        elif "native" in h_low:
                            nat_i = i
                    if rom_i is None or lang_i is None:
                        continue
                    for line in lines[1:]:
                        parts = line.split(sep)
                        if max(rom_i, lang_i) >= len(parts):
                            continue
                        lang_name = parts[lang_i].strip()
                        rom       = parts[rom_i].strip()
                        code      = LANG_NAME_TO_CODE.get(lang_name)
                        if code and rom:
                            lang_rom_sentences.setdefault(code, []).append(rom)

            # If no structured data found, try plain text per language file
            if not lang_rom_sentences:
                print("  No structured data found. Trying per-language files …")
                for fname in all_files:
                    low = fname.lower()
                    if not low.endswith(".txt"):
                        continue
                    # Extract language from filename path
                    parts = fname.replace("\\", "/").split("/")
                    # e.g., 'romanized/hin/test.txt'
                    for p in parts:
                        code = LANG_NAME_TO_CODE.get(p) or (p if len(p) == 2 else None)
                        if code in LANG_NAME_TO_CODE.values():
                            with zf.open(fname) as f:
                                texts = [l.strip() for l in
                                         f.read().decode("utf-8", errors="replace").splitlines()
                                         if l.strip()]
                            lang_rom_sentences.setdefault(code, []).extend(texts)
                            break

    except zipfile.BadZipFile as e:
        print(f"  ERROR: Bad ZIP: {e}")
        return

    print(f"\n  Languages found — Romanized: {list(lang_rom_sentences.keys())}")
    print(f"  Languages found — Native:    {list(lang_nat_sentences.keys())}")

    for code, sentences in lang_rom_sentences.items():
        out_dir = os.path.join(save_dir, "bhasha_abhijnaanam", code)
        save_text_to_file(sentences, os.path.join(out_dir, "test.txt"))

    for code, sentences in lang_nat_sentences.items():
        out_dir = os.path.join(save_dir, "bhasha_abhijnaanam_native", code)
        save_text_to_file(sentences, os.path.join(out_dir, "test.txt"))

    print("\n  Bhasha-Abhijnaanam download complete.")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Dataset Downloader     ║")
    print("╚══════════════════════════════════════════╝\n")

    os.makedirs(RAW_NATIVE_DIR,    exist_ok=True)
    os.makedirs(RAW_ROMANIZED_DIR, exist_ok=True)

    # ── 1: Native scripts + English from IndicCorpV2 ──────────
    download_indiccorpv2(
        save_dir=RAW_NATIVE_DIR,
        train_per_lang=50_000,
        test_per_lang=30_000,
    )

    # ── 2: Romanized word pairs from Aksharantar ──────────────
    download_aksharantar(save_dir=RAW_ROMANIZED_DIR)

    # ── 3: Romanized LID sentences (Bhasha-Abhijnaanam) ───────
    download_bhasha_abhijnaanam(save_dir=RAW_ROMANIZED_DIR)

    print("\n✅ All datasets downloaded.")
    print(f"  Native    → {RAW_NATIVE_DIR}")
    print(f"  Romanized → {RAW_ROMANIZED_DIR}")
    print("\nNext step: Run  python 02_preprocess_native.py")
