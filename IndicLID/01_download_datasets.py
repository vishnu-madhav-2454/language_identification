# ==============================================================
# 01_download_datasets.py
#
# STEP 1 — Download all datasets used in the paper:
#
#   1. IndicCorpV2 (ai4bharat/IndicCorpV2)
#      → Native-script sentences for 22 Indian languages + English
#      → 100k sentences per language for training
#
#   2. Bhasha-Abhijnaanam (ai4bharat/Bhasha-Abhijnaanam)
#      → Benchmark test set (native + romanized)
#      → Used ONLY for evaluation (never training)
#
# Data is streamed to avoid downloading entire multi-GB files.
#
# Run:  python 01_download_datasets.py
# ==============================================================

import io
import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_NATIVE_DIR,
    RAW_ROMANIZED_DIR,
    PROCESSED_DIR,
    BENCHMARK_NATIVE_CSV,
    BENCHMARK_ROMANIZED_CSV,
    INDICCORPV2_FILE_CODES,
    INDIC_LANGUAGES,
    ISO_TO_NATIVE_LABEL,
    ISO_TO_ROMAN_LABEL,
    TRAINING_SAMPLES_PER_LANG,
)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def save_lines(lines: list, filepath: str):
    """Save list of strings (one per line) to a UTF-8 text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.strip() + "\n")
    print(f"      Saved {len(lines):,} lines → {filepath}")


def stream_url_lines(
    url: str,
    n_lines: int,
    timeout: int = 120,
    max_retries: int = 5,
    retry_delay: float = 5.0,
) -> list:
    """
    Stream the first n_lines non-empty lines from a plain-text URL.
    Retries up to max_retries times on network errors (connection drops
    are common with large HuggingFace files).
    This avoids downloading entire multi-GB corpus files.
    """
    import time

    lines = []
    for attempt in range(1, max_retries + 1):
        lines = []
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=False):
                    # Decode manually as UTF-8 to avoid requests falling
                    # back to ISO-8859-1 for text/* content (RFC 2616 §3.7.1)
                    if raw:
                        text = raw.decode("utf-8", errors="replace").strip()
                        if text:
                            lines.append(text)
                    if len(lines) >= n_lines:
                        break
            # If we got here without exception, we're done
            return lines
        except requests.exceptions.HTTPError as e:
            print(f"      HTTP {e.response.status_code}: {url}")
            return []   # 404/403 won't be fixed by retrying
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait = retry_delay * attempt
                print(f"      Network error (attempt {attempt}/{max_retries}): {e}")
                print(f"      Retrying in {wait:.0f}s …")
                time.sleep(wait)
            else:
                print(f"      Network error after {max_retries} attempts: {e}")
    return lines


# ─────────────────────────────────────────────────────────────────
# DATASET 1 — IndicCorpV2 (Native Script + English)
#
# URL pattern:
#   https://huggingface.co/datasets/ai4bharat/IndicCorpV2/
#   resolve/main/data/<lang_code>.txt
#
# Paper methodology:
#   • 100k sentences per language for training
#   • Sentences normalized via IndicNLP library
#   • Oversampling for low-resource languages
#
# We stream first (100k + extra buffer) lines per language to get
# ~100k clean sentences after preprocessing.
# ─────────────────────────────────────────────────────────────────
BASE_INDICCORPV2 = (
    "https://huggingface.co/datasets/ai4bharat/IndicCorpV2"
    "/resolve/main/data"
)

# All language files verified at:
# https://huggingface.co/datasets/ai4bharat/IndicCorpV2/tree/main/data
# Correct codes are stored in config.py INDICCORPV2_FILE_CODES.


def download_indiccorpv2(
    save_dir: str,
    n_per_lang: int = TRAINING_SAMPLES_PER_LANG,
):
    """
    Stream native-script sentences from IndicCorpV2.
    Saves to: save_dir/<iso_code>/train.txt
    """
    print("\n" + "=" * 65)
    print("  DOWNLOADING: IndicCorpV2 (Native Script sentences)")
    print("=" * 65)

    # We need a bit more than n_per_lang to account for rejected lines
    buffer = int(n_per_lang * 1.3)

    all_langs = dict(INDICCORPV2_FILE_CODES)

    for iso_code, file_code in tqdm(all_langs.items(), desc="  Languages"):
        out_path = os.path.join(save_dir, iso_code, "train.txt")

        if os.path.exists(out_path):
            existing = sum(1 for _ in open(out_path, encoding="utf-8"))
            if existing >= n_per_lang // 2:
                print(f"  [{iso_code}] Already exists ({existing:,} lines). Skipping.")
                continue

        url = f"{BASE_INDICCORPV2}/{file_code}.txt"
        print(f"\n  [{iso_code}] Streaming {file_code}.txt …")

        lines = stream_url_lines(url, buffer)

        if not lines:
            print(f"  [{iso_code}] WARNING: No data fetched. "
                  f"Language may not be in IndicCorpV2 or URL changed.")
            # Try creating a placeholder for low-resource languages
            # so the pipeline can still run
            _create_placeholder(save_dir, iso_code)
            continue

        # Take up to n_per_lang lines
        lines = lines[:n_per_lang]
        print(f"  [{iso_code}] Fetched {len(lines):,} lines.")
        save_lines(lines, out_path)

    print("\n  IndicCorpV2 download complete.")


def _create_placeholder(save_dir: str, iso_code: str):
    """
    Create a small placeholder file for a missing language
    so the pipeline does not crash. The preprocessor will skip
    it gracefully.
    """
    out_path = os.path.join(save_dir, iso_code, "train.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write an empty file as a marker
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("")
    print(f"  [{iso_code}] Placeholder created at: {out_path}")


# ─────────────────────────────────────────────────────────────────
# DATASET 2 — Bhasha-Abhijnaanam (LID Benchmark)
#
# HuggingFace: ai4bharat/Bhasha-Abhijnaanam
# Single ZIP: bhasha-abhijnaanam.zip
#
# Contains the benchmark test set described in the paper:
#   • Section 2: Native-script test set from FLORES-200 + Dakshina
#     + new translations for Bodo, Konkani, Dogri, Manipuri
#   • Section 2.2: Romanized test set (Dakshina filtered + new)
#
# This is ONLY used for evaluation, never for training.
# The paper was careful to ensure no test/train overlap.
# ─────────────────────────────────────────────────────────────────

BHASHA_ZIP_URL = (
    "https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam"
    "/resolve/main/bhasha-abhijnaanam.zip"
)

# The benchmark uses slightly different language codes
BHASHA_LANG_MAP = {
    # Full name → iso_code
    "Assamese":  "as",  "Bengali":   "bn",  "Bodo":       "brx",
    "Dogri":     "doi", "Gujarati":  "gu",  "Hindi":      "hi",
    "Kannada":   "kn",  "Kashmiri":  "ks",  "Konkani":    "kok",
    "Maithili":  "mai", "Malayalam": "ml",  "Manipuri":   "mni",
    "Marathi":   "mr",  "Nepali":    "ne",  "Oriya":      "or",
    "Punjabi":   "pa",  "Sanskrit":  "sa",  "Santali":    "sat",
    "Sindhi":    "sd",  "Tamil":     "ta",  "Telugu":     "te",
    "Urdu":      "ur",
    # Also handles 3-letter codes that appear in the dataset
    "asm":  "as", "ben": "bn",  "brx": "brx", "doi": "doi",
    "guj":  "gu", "hin": "hi",  "kan": "kn",  "kas": "ks",
    "kok":  "kok","mai": "mai", "mal": "ml",  "mni": "mni",
    "mar":  "mr", "nep": "ne",  "ori": "or",  "pan": "pa",
    "san":  "sa", "sat": "sat", "snd": "sd",  "tam": "ta",
    "tel":  "te", "urd": "ur",
}


def download_bhasha_abhijnaanam(processed_dir: str):
    """
    Download and extract the Bhasha-Abhijnaanam benchmark.
    Creates benchmark_native.csv and benchmark_romanized.csv
    in the processed_dir.
    """
    import pandas as pd

    print("\n" + "=" * 65)
    print("  DOWNLOADING: Bhasha-Abhijnaanam (LID Benchmark Test Set)")
    print("=" * 65)

    native_out   = os.path.join(processed_dir, "benchmark_native.csv")
    romanized_out= os.path.join(processed_dir, "benchmark_romanized.csv")

    if os.path.exists(native_out) and os.path.exists(romanized_out):
        print("  Benchmark files already exist. Skipping download.")
        n_native   = len(pd.read_csv(native_out))
        n_romanized= len(pd.read_csv(romanized_out))
        print(f"  Native benchmark   : {n_native:,} rows")
        print(f"  Romanized benchmark: {n_romanized:,} rows")
        return

    print(f"  Downloading: {BHASHA_ZIP_URL}")
    try:
        resp = requests.get(BHASHA_ZIP_URL, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        print("  You can manually download the dataset from:")
        print("  https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam")
        return

    print(f"  Downloaded {len(resp.content):,} bytes. Extracting …")
    os.makedirs(processed_dir, exist_ok=True)

    native_rows   = []
    romanized_rows= []

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            all_files = zf.namelist()
            print(f"  ZIP contains {len(all_files)} files.")

            for fname in sorted(all_files):
                low  = fname.lower().replace("\\", "/")
                name = fname.replace("\\", "/")

                if not (low.endswith(".tsv") or low.endswith(".csv")
                        or low.endswith(".json") or low.endswith(".jsonl")):
                    continue

                with zf.open(fname) as f:
                    content = f.read().decode("utf-8", errors="replace")

                # ── Try to parse as JSON / JSONL ──
                if low.endswith((".json", ".jsonl")):
                    _parse_json_file(content, fname, native_rows, romanized_rows)
                    continue

                # ── Try to parse as TSV / CSV ──
                sep = "\t" if low.endswith(".tsv") else ","
                lines = content.splitlines()
                if not lines:
                    continue

                header = [h.strip().lower() for h in lines[0].split(sep)]
                _parse_tabular_file(
                    lines, header, sep, fname,
                    native_rows, romanized_rows
                )

    except zipfile.BadZipFile as e:
        print(f"  ERROR: Bad ZIP: {e}")
        return

    print(f"\n  Parsed {len(native_rows):,} native rows, "
          f"{len(romanized_rows):,} romanized rows.")

    if native_rows:
        df_native = pd.DataFrame(native_rows)
        df_native.to_csv(native_out, index=False, encoding="utf-8")
        print(f"  Saved native benchmark → {native_out}")

    if romanized_rows:
        df_roman = pd.DataFrame(romanized_rows)
        df_roman.to_csv(romanized_out, index=False, encoding="utf-8")
        print(f"  Saved romanized benchmark → {romanized_out}")

    if not native_rows and not romanized_rows:
        print("  WARNING: No benchmark data extracted.")
        print("  Expected JSON / TSV / CSV files inside the ZIP.")
        print("  Please check the ZIP structure manually.")
        _create_benchmark_from_hf_api(processed_dir)


def _resolve_iso(raw_lang: str) -> str:
    """Map a raw language string (any format) to our iso_code."""
    raw = raw_lang.strip()
    # Direct match
    if raw in BHASHA_LANG_MAP:
        return BHASHA_LANG_MAP[raw]
    # Try title-case
    if raw.title() in BHASHA_LANG_MAP:
        return BHASHA_LANG_MAP[raw.title()]
    # Try lower-case
    if raw.lower() in BHASHA_LANG_MAP:
        return BHASHA_LANG_MAP[raw.lower()]
    return None


def _parse_json_file(content, fname, native_rows, romanized_rows):
    """Parse a JSON or JSONL benchmark file."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try JSONL
        data = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Could be {"lang": [records...]} or {"id": record}
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, list):
            for lang_key, items in data.items():
                for item in (items if isinstance(items, list) else [items]):
                    if isinstance(item, dict) and "language" not in item:
                        item = {**item, "language": lang_key}
                    records.append(item)
        elif isinstance(first_val, dict):
            records = list(data.values())

    for rec in records:
        if not isinstance(rec, dict):
            continue
        lang_raw = (rec.get("language") or rec.get("lang") or "").strip()
        native   = (rec.get("native sentence") or rec.get("native_sentence") or
                    rec.get("sentence") or "").strip()
        roman    = (rec.get("romanized sentence") or rec.get("romanized_sentence") or
                    rec.get("romanized") or "").strip()
        iso = _resolve_iso(lang_raw)
        if not iso:
            continue
        if native:
            native_label = ISO_TO_NATIVE_LABEL.get(iso, -1)
            if native_label >= 0:
                native_rows.append({
                    "sentence": native, "label": native_label, "iso": iso
                })
        if roman:
            roman_label = ISO_TO_ROMAN_LABEL.get(iso, -1)
            if roman_label >= 0:
                romanized_rows.append({
                    "sentence": roman, "label": roman_label, "iso": iso
                })


def _parse_tabular_file(lines, header, sep, fname, native_rows, romanized_rows):
    """Parse a TSV / CSV benchmark file."""
    # Find column indices
    col_map = {}
    for i, h in enumerate(header):
        if any(k in h for k in ("native", "original")):
            col_map.setdefault("native", i)
        if any(k in h for k in ("roman", "latin")):
            col_map.setdefault("roman", i)
        if "lang" in h:
            col_map.setdefault("lang", i)

    if "lang" not in col_map:
        return  # can't determine language

    for line in lines[1:]:
        parts = line.split(sep)
        if not parts:
            continue
        lang_raw = ""
        if col_map.get("lang") is not None and col_map["lang"] < len(parts):
            lang_raw = parts[col_map["lang"]].strip().strip('"')
        iso = _resolve_iso(lang_raw)
        if not iso:
            continue

        native = ""
        if col_map.get("native") is not None and col_map["native"] < len(parts):
            native = parts[col_map["native"]].strip().strip('"')
        roman = ""
        if col_map.get("roman") is not None and col_map["roman"] < len(parts):
            roman  = parts[col_map["roman"]].strip().strip('"')

        if native:
            nl = ISO_TO_NATIVE_LABEL.get(iso, -1)
            if nl >= 0:
                native_rows.append({"sentence": native, "label": nl, "iso": iso})
        if roman:
            rl = ISO_TO_ROMAN_LABEL.get(iso, -1)
            if rl >= 0:
                romanized_rows.append({"sentence": roman, "label": rl, "iso": iso})


def _create_benchmark_from_hf_api(processed_dir: str):
    """
    Fallback: try to get benchmark data via the HuggingFace datasets API.
    This handles cases where the ZIP structure changed.
    """
    import pandas as pd
    print("\n  Trying HuggingFace API fallback …")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/Bhasha-Abhijnaanam", trust_remote_code=True)
        print(f"  Loaded dataset splits: {list(ds.keys())}")

        native_rows, roman_rows = [], []
        for split_name, split_data in ds.items():
            for item in tqdm(split_data, desc=f"  Processing {split_name}"):
                lang_raw = str(item.get("language","") or item.get("lang","")).strip()
                iso = _resolve_iso(lang_raw)
                if not iso:
                    continue
                for field in ("native sentence", "native_sentence", "sentence_native"):
                    native = item.get(field, "")
                    if native:
                        nl = ISO_TO_NATIVE_LABEL.get(iso, -1)
                        if nl >= 0:
                            native_rows.append({"sentence": native, "label": nl, "iso": iso})
                        break
                for field in ("romanized sentence", "romanized_sentence", "sentence_romanized"):
                    roman = item.get(field, "")
                    if roman:
                        rl = ISO_TO_ROMAN_LABEL.get(iso, -1)
                        if rl >= 0:
                            roman_rows.append({"sentence": roman, "label": rl, "iso": iso})
                        break

        if native_rows:
            pd.DataFrame(native_rows).to_csv(
                os.path.join(processed_dir, "benchmark_native.csv"),
                index=False, encoding="utf-8"
            )
            print(f"  API fallback: saved {len(native_rows):,} native rows.")
        if roman_rows:
            pd.DataFrame(roman_rows).to_csv(
                os.path.join(processed_dir, "benchmark_romanized.csv"),
                index=False, encoding="utf-8"
            )
            print(f"  API fallback: saved {len(roman_rows):,} romanized rows.")

    except Exception as e:
        print(f"  API fallback failed: {e}")
        print("  Please download the benchmark manually from:")
        print("  https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam")


# ─────────────────────────────────────────────────────────────────
# DATASET 3 — Supplemental data from Wikimedia Wikipedia + CC-100
#
# Paper references: "IndicCorp, NLLB, Wikipedia, Vikaspedia" (Section 3.1)
#
# We supplement IndicCorpV2 for languages below the 100k target using:
#   1. Wikimedia Wikipedia (wikimedia/wikipedia on HuggingFace)
#   2. CC-100 CommonCrawl (rahular/varta via HuggingFace)
#
# Languages prioritised: doi, ks, sd, ur, mni (below 70k after Step 2)
# ─────────────────────────────────────────────────────────────────

# HuggingFace Wikipedia edition codes for Indic languages
# Source: https://huggingface.co/datasets/wikimedia/wikipedia
WIKI_EDITION_CODES = {
    "as":  "20231101.as",
    "bn":  "20231101.bn",
    "gu":  "20231101.gu",
    "hi":  "20231101.hi",
    "kn":  "20231101.kn",
    "kok": None,               # Konkani: no Wikipedia edition in HuggingFace
    "ks":  "20231101.ks",
    "mai": "20231101.mai",
    "ml":  "20231101.ml",
    "mni": None,               # Meetei: not in HuggingFace Wikipedia
    "mr":  "20231101.mr",
    "ne":  "20231101.ne",
    "or":  "20231101.or",
    "pa":  "20231101.pa",
    "sa":  "20231101.sa",
    "sd":  "20231101.sd",
    "ta":  "20231101.ta",
    "te":  "20231101.te",
    "ur":  "20231101.ur",
    "doi": None,               # Dogri: no Wikipedia edition
    "brx": None,               # Bodo:  no Wikipedia edition
}


def download_supplemental_wikipedia(
    save_dir: str,
    n_per_lang: int = TRAINING_SAMPLES_PER_LANG,
    threshold: float = 0.85,
):
    """
    Supplement native-script train files with Wikipedia text for languages
    that have fewer than threshold * n_per_lang lines.

    Uses `wikimedia/wikipedia` dataset via HuggingFace `datasets` library
    (streaming=True to avoid downloading entire dumps).

    Parameters
    ----------
    save_dir   : same as RAW_NATIVE_DIR used in download_indiccorpv2
    n_per_lang : target number of lines per language (default: 100_000)
    threshold  : fraction of n_per_lang below which we supplement (default 0.85)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [WIKI] 'datasets' library not installed. Skipping supplemental download.")
        return

    print("\n" + "=" * 65)
    print("  SUPPLEMENTAL: Wikimedia Wikipedia (for low-resource langs)")
    print("=" * 65)

    target_min = int(n_per_lang * threshold)

    for iso_code, wiki_edition in WIKI_EDITION_CODES.items():
        train_file = os.path.join(save_dir, iso_code, "train.txt")
        if not os.path.exists(train_file):
            continue

        current = sum(1 for _ in open(train_file, encoding="utf-8", errors="replace"))
        if current >= target_min:
            print(f"  [{iso_code}] {current:,} lines — already at/above threshold. Skipping.")
            continue

        if wiki_edition is None:
            print(f"  [{iso_code}] {current:,} lines — no Wikipedia edition available.")
            continue

        needed = n_per_lang - current
        print(f"  [{iso_code}] {current:,} lines → need {needed:,} more from Wikipedia "
              f"({wiki_edition}) …")

        try:
            ds = load_dataset(
                "wikimedia/wikipedia", wiki_edition,
                split="train", streaming=True, trust_remote_code=True,
            )
            extra_lines = []
            for item in ds:
                text = item.get("text", "")
                for line in text.split("\n"):
                    line = line.strip()
                    if len(line) >= 30:          # skip very short lines / section headers
                        extra_lines.append(line)
                    if len(extra_lines) >= needed * 1.5:
                        break
                if len(extra_lines) >= needed * 1.5:
                    break

            if extra_lines:
                extra_lines = extra_lines[:needed]
                with open(train_file, "a", encoding="utf-8") as f:
                    for line in extra_lines:
                        f.write(line + "\n")
                print(f"  [{iso_code}] Appended {len(extra_lines):,} Wikipedia lines "
                      f"→ total now {current + len(extra_lines):,}")
            else:
                print(f"  [{iso_code}] No usable lines found in Wikipedia edition.")

        except Exception as e:
            print(f"  [{iso_code}] Wikipedia download failed: {e}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║  IndicLID (Bhasha-Abhijnaanam) — Dataset Downloader  ║")
    print("║  ACL 2023 | AI4Bharat                                ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    os.makedirs(RAW_NATIVE_DIR,   exist_ok=True)
    os.makedirs(RAW_ROMANIZED_DIR,exist_ok=True)
    os.makedirs(PROCESSED_DIR,    exist_ok=True)

    # ── 1. Native corpus from IndicCorpV2 ──
    download_indiccorpv2(
        save_dir=RAW_NATIVE_DIR,
        n_per_lang=TRAINING_SAMPLES_PER_LANG,
    )

    # ── 2. Bhasha-Abhijnaanam benchmark ──
    download_bhasha_abhijnaanam(processed_dir=PROCESSED_DIR)

    # ── 3. Supplemental Wikipedia data (paper: Wikipedia as extra source) ──
    #    Downloads only for languages below 85% of the 100k target.
    #    Safe to skip if connection is slow — Step 2 handles low-resource gracefully.
    download_supplemental_wikipedia(
        save_dir=RAW_NATIVE_DIR,
        n_per_lang=TRAINING_SAMPLES_PER_LANG,
        threshold=0.85,
    )

    print("\n" + "=" * 65)
    print("  Downloads complete.")
    print(f"  Native raw data   → {RAW_NATIVE_DIR}")
    print(f"  Benchmark data    → {PROCESSED_DIR}")
    print("\n  Next step: python 02_preprocess_native.py")
    print("=" * 65)
