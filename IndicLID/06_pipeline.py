# ==============================================================
# 06_pipeline.py
#
# STEP 6 — IndicLID Ensemble Pipeline
#
# Paper (Section 3.4 + Figure 1):
#
#   The IndicLID classifier is a PIPELINE of multiple classifiers:
#
#   Step 1: Detect script type
#     → Count roman characters in the input
#     → If >50% roman: invoke IndicLID-FTR (romanized linear classifier)
#     → Else: invoke IndicLID-FTN (native script linear classifier)
#
#   Step 2: Confidence-based escalation (romanized only)
#     → If IndicLID-FTR probability score > 0.6: return FTR result
#     → If probability ≤ 0.6: invoke IndicLID-BERT (more accurate)
#
#   This gives the best speed-accuracy trade-off (Table 9):
#     threshold=0.6 → 80.40% accuracy, ~10 sentences/second
#
#   Paper comparison:
#     IndicLID-FTR alone  : 71.49% acc, 37,037 sent/s
#     IndicLID-BERT alone : 80.04% acc,      3 sent/s
#     IndicLID (ensemble) : 80.40% acc,     10 sent/s  ← BEST TRADE-OFF
#
# Run:
#   python 06_pipeline.py
#   python 06_pipeline.py --text "vanakkam naan nalama irukkiren"
#   python 06_pipeline.py --text "नमस्ते आप कैसे हैं"
# ==============================================================

import os
import re
import sys
import json
import string
import numpy as np
import torch
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FTN_MODEL_PATH,
    FTR_MODEL_PATH,
    BERT_MODEL_DIR,
    ROMAN_CHAR_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    MAX_TOKEN_LENGTH,
    LABEL_TO_ID,
    ID_TO_LABEL,
    ROMANIZED_LABEL_IDS,
    NATIVE_LABEL_IDS,
    ENGLISH_LABEL_ID,
    OTHERS_LABEL_ID,
    INDIC_LANGUAGES,
)


# ─────────────────────────────────────────────────────────────────
# SCRIPT DETECTION
# ─────────────────────────────────────────────────────────────────

def compute_roman_fraction(text: str) -> float:
    """
    Compute the fraction of alphabetic characters that are in the
    Roman/Latin script (a-z, A-Z).

    Paper: "Whether 50% of the char in sentence are Roman?"
    """
    total_alpha = 0
    roman_alpha = 0
    for ch in text:
        if ch.isalpha():
            total_alpha += 1
            if ch in string.ascii_letters:
                roman_alpha += 1
    if total_alpha == 0:
        return 0.0
    return roman_alpha / total_alpha


# ─────────────────────────────────────────────────────────────────
# FASTTEXT MODEL LOADER
# ─────────────────────────────────────────────────────────────────

class FastTextClassifier:
    """
    Wraps a FastText supervised model for IndicLID.
    Provides predict() with confidence scores.
    """

    def __init__(self, model_path: str, name: str):
        self.name = name
        self._model = None
        self._path  = model_path
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            print(f"  [{self.name}] Model not found: {self._path}")
            print(f"  Run 04_train_fasttext.py first.")
            return
        try:
            import fasttext
            self._model = fasttext.load_model(self._path)
            print(f"  [{self.name}] Loaded: {self._path}")
        except ImportError:
            print(f"  [{self.name}] fasttext not installed. "
                  f"Run: pip install fasttext-wheel (Windows) or fasttext (Linux)")
        except Exception as e:
            print(f"  [{self.name}] Load error: {e}")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, text: str, k: int = 1) -> tuple:
        """
        Predict the language of `text`.

        Returns
        -------
        (label_str, confidence)
          label_str  : e.g. "hi_Deva" or "ta_Latn"
          confidence : float in [0, 1]
        """
        if not self._model:
            return ("xx", 0.0)

        # FastText returns a tuple: ([labels], [scores])
        labels, scores = self._model.predict(text, k=k)
        if not labels:
            return ("xx", 0.0)

        # Remove the "__label__" prefix
        label = labels[0].replace("__label__", "")
        score = float(scores[0])
        return (label, score)

    def predict_batch(self, texts: list, k: int = 1) -> list:
        """
        Predict for a list of texts.
        Returns list of (label_str, confidence) tuples.
        """
        if not self._model:
            return [("xx", 0.0)] * len(texts)
        results = []
        for text in texts:
            results.append(self.predict(text, k=k))
        return results


# ─────────────────────────────────────────────────────────────────
# INDICBERT LOADER
# ─────────────────────────────────────────────────────────────────

class IndicBERTClassifier:
    """
    Wraps the fine-tuned IndicBERT model for romanized text LID.
    Used as the fallback when IndicLID-FTR confidence is low.
    """

    def __init__(self, model_dir: str):
        self._model     = None
        self._tokenizer = None
        self._dir       = model_dir
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._id2label  = {}    # local_id → label string (e.g. "hi_Latn")
        self._load()

    def _load(self):
        if not os.path.isdir(self._dir):
            print(f"  [BERT] Model not found: {self._dir}")
            print(f"  Run 05_train_indicbert.py first.")
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"  [BERT] Loading tokenizer from: {self._dir}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._dir)
            print(f"  [BERT] Loading model from: {self._dir}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._dir
            ).to(self._device).eval()
            print(f"  [BERT] Loaded on {self._device}")

            # Load label mapping
            mapping_path = os.path.join(self._dir, "label_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path) as f:
                    m = json.load(f)
                self._id2label = {int(k): v for k, v in m["id2label"].items()}
            else:
                # Fallback: assume labels are romanized IDs 24-44 in order
                sorted_roman = sorted(ROMANIZED_LABEL_IDS)
                self._id2label = {
                    i: ID_TO_LABEL[g]
                    for i, g in enumerate(sorted_roman)
                }
        except Exception as e:
            print(f"  [BERT] Load error: {e}")

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, text: str) -> tuple:
        """
        Predict the romanized language of `text`.

        Returns
        -------
        (label_str, confidence)
        """
        if not self._model:
            return ("xx", 0.0)

        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        label_str  = self._id2label.get(pred_id, "xx")
        return (label_str, confidence)

    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        Predict for a batch of texts.
        Returns list of (label_str, confidence) tuples.
        """
        if not self._model:
            return [("xx", 0.0)] * len(texts)

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for j in range(len(batch)):
                pred_id    = int(np.argmax(probs[j]))
                confidence = float(probs[j][pred_id])
                label_str  = self._id2label.get(pred_id, "xx")
                results.append((label_str, confidence))
        return results


# ─────────────────────────────────────────────────────────────────
# MAIN INDICLID PIPELINE (Figure 1)
# ─────────────────────────────────────────────────────────────────

class IndicLIDPipeline:
    """
    IndicLID Ensemble Pipeline (Figure 1 of the paper).

    Implements the three-stage approach:
    1. Script detection (>50% roman → romanized path, else native path)
    2. Fast linear classifier (IndicLID-FTR or IndicLID-FTN)
    3. High-confidence FTR → return; Low-confidence FTR → escalate to BERT
    """

    def __init__(
        self,
        roman_threshold: float = ROMAN_CHAR_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_bert: bool = True,
    ):
        self.roman_threshold      = roman_threshold
        self.confidence_threshold = confidence_threshold
        self.use_bert             = use_bert

        print("  Loading IndicLID models …")
        self.ftn  = FastTextClassifier(FTN_MODEL_PATH, "FTN")
        self.ftr  = FastTextClassifier(FTR_MODEL_PATH, "FTR")
        self.bert = IndicBERTClassifier(BERT_MODEL_DIR) if use_bert else None
        print("  Models ready.\n")

    def identify(self, text: str, verbose: bool = False) -> dict:
        """
        Identify the language of a text sample.

        Parameters
        ----------
        text    : input text (single sentence or word)
        verbose : print pipeline steps

        Returns
        -------
        dict with:
            label     : label string (e.g. "hi_Deva", "ta_Latn")
            label_id  : integer label ID
            confidence: float
            model_used: which model gave the final result
            is_roman  : whether input was treated as romanized
        """
        if not text.strip():
            return {
                "label": "xx", "label_id": OTHERS_LABEL_ID,
                "confidence": 0.0, "model_used": "none", "is_roman": False
            }

        # ── Step 1: Script detection ──
        roman_frac = compute_roman_fraction(text)
        is_roman   = roman_frac > self.roman_threshold

        if verbose:
            print(f"  Input       : '{text}'")
            print(f"  Roman frac  : {roman_frac:.2f}  →  {'ROMANIZED' if is_roman else 'NATIVE'} path")

        # ── Step 2: Apply the correct branch ──
        if not is_roman:
            # ── Native path: IndicLID-FTN ──
            if self.ftn.is_loaded():
                label_str, confidence = self.ftn.predict(text)
                label_id = LABEL_TO_ID.get(label_str, OTHERS_LABEL_ID)
                if verbose:
                    print(f"  IndicLID-FTN: {label_str}  (conf={confidence:.3f})")
                return {
                    "label": label_str, "label_id": label_id,
                    "confidence": confidence, "model_used": "FTN",
                    "is_roman": False
                }
            else:
                if verbose:
                    print("  FTN not loaded. Returning 'xx'.")
                return {
                    "label": "xx", "label_id": OTHERS_LABEL_ID,
                    "confidence": 0.0, "model_used": "none", "is_roman": False
                }

        else:
            # ── Romanized path: IndicLID-FTR first ──
            label_str_ftr, confidence_ftr = ("xx", 0.0)
            if self.ftr.is_loaded():
                label_str_ftr, confidence_ftr = self.ftr.predict(text)
                if verbose:
                    print(f"  IndicLID-FTR: {label_str_ftr}  (conf={confidence_ftr:.3f})")

            # ── Step 3: Confidence check ──
            if confidence_ftr >= self.confidence_threshold:
                # FTR is confident → return its result
                label_id = LABEL_TO_ID.get(label_str_ftr, OTHERS_LABEL_ID)
                if verbose:
                    print(f"  Confidence {confidence_ftr:.3f} ≥ {self.confidence_threshold} → FTR result used")
                return {
                    "label": label_str_ftr, "label_id": label_id,
                    "confidence": confidence_ftr, "model_used": "FTR",
                    "is_roman": True
                }
            else:
                # FTR not confident → escalate to IndicLID-BERT
                if verbose:
                    print(f"  Confidence {confidence_ftr:.3f} < {self.confidence_threshold} → escalating to BERT")

                if self.bert and self.bert.is_loaded():
                    label_str_bert, confidence_bert = self.bert.predict(text)
                    label_id = LABEL_TO_ID.get(label_str_bert, OTHERS_LABEL_ID)
                    if verbose:
                        print(f"  IndicLID-BERT: {label_str_bert}  (conf={confidence_bert:.3f})")
                    return {
                        "label": label_str_bert, "label_id": label_id,
                        "confidence": confidence_bert, "model_used": "BERT",
                        "is_roman": True
                    }
                else:
                    # BERT not available, fall back to FTR result
                    label_id = LABEL_TO_ID.get(label_str_ftr, OTHERS_LABEL_ID)
                    if verbose:
                        print("  BERT not loaded. Using FTR result.")
                    return {
                        "label": label_str_ftr, "label_id": label_id,
                        "confidence": confidence_ftr, "model_used": "FTR-fallback",
                        "is_roman": True
                    }

    def identify_batch(
        self, texts: list, verbose: bool = False
    ) -> list:
        """
        Identify languages for a batch of texts.
        Returns list of result dicts (same as identify()).
        """
        return [self.identify(t, verbose=verbose) for t in texts]

    def get_language_name(self, label_str: str) -> str:
        """Convert label string to human-readable language name."""
        if label_str == "en":
            return "English"
        if label_str == "xx":
            return "Others/Unknown"
        # Split "hi_Deva" → iso="hi", script="Deva"
        parts = label_str.split("_")
        iso   = parts[0]
        info  = INDIC_LANGUAGES.get(iso)
        if info:
            lang_name = info[0]
            script    = info[1]
            suffix    = "_Latn" if label_str.endswith("_Latn") else f"_{script[:4]}"
            return f"{lang_name} ({'Romanized' if '_Latn' in label_str else script})"
        return label_str


# ─────────────────────────────────────────────────────────────────
# THRESHOLD ANALYSIS (Appendix C replication — Table 9)
# ─────────────────────────────────────────────────────────────────

def analyze_thresholds(pipeline: IndicLIDPipeline, test_file: Optional[str] = None):
    """
    Replicate Table 9 from the paper:
    Evaluate different confidence thresholds on the test set.

    Table 9 (from the paper):
    threshold | P     | R     | F1    | Acc   | Throughput
    0.1       | 63.13 | 78.02 | 63.29 | 71.49 | 50,000
    0.3       | 65.50 | 79.64 | 66.15 | 73.84 | 54
    0.6       | 72.74 | 84.51 | 74.72 | 80.40 | 10    ← chosen
    0.9       | 73.51 | 84.50 | 75.35 | 80.62 | 6
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  Threshold analysis (replicating Table 9) …")
    print(f"  {'Thresh':>8}  {'BERT calls':>12}  {'Total':>8}")
    print(f"  {'─'*40}")

    for thr in thresholds:
        pipeline.confidence_threshold = thr
        print(f"  {thr:>8.1f}  — (run evaluate.py for full stats)")

    # Reset to default
    pipeline.confidence_threshold = CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────

DEMO_SENTENCES = [
    # Native scripts
    ("नमस्ते आप कैसे हैं",          "hi_Deva  (Hindi, Devanagari)"),
    ("আমি ভালো আছি তুমি কেমন আছো",  "bn_Beng  (Bengali)"),
    ("வணக்கம் நான் நலமாக இருக்கிறேன்", "ta_Taml  (Tamil)"),
    ("ناماستے آپ کیسے ہیں",          "ur_Arab  (Urdu, Perso-Arabic)"),
    ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਕਿਵੇਂ ਹੋ",      "pa_Guru  (Punjabi, Gurmukhi)"),
    # Romanized scripts
    ("namaste aap kaise hain",         "hi_Latn  (Hindi Romanized)"),
    ("ami bhalo achi tumi kemon acho", "bn_Latn  (Bengali Romanized)"),
    ("vanakkam naan nalama irukkiren", "ta_Latn  (Tamil Romanized)"),
    ("aap ek dum sahi bol rahe ho",    "hi_Latn  (Hindi Romanized)"),
    # English
    ("The quick brown fox",            "en       (English)"),
]


def run_demo(pipeline: IndicLIDPipeline):
    """Run the demo with example sentences from the paper."""
    print(f"\n  {'─'*70}")
    print(f"  {'Input':40s}  {'Predicted':18s}  {'Conf':6s}  {'Model'}")
    print(f"  {'─'*70}")

    for text, expected in DEMO_SENTENCES:
        result = pipeline.identify(text)
        label_nice = pipeline.get_language_name(result["label"])
        conf       = result["confidence"]
        model      = result["model_used"]
        correct    = "✓" if result["label"].split("_")[0] in expected else "≈"
        print(f"  {text[:38]:40s}  {result['label']:18s}  {conf:6.3f}  [{model}] {correct}")

    print(f"  {'─'*70}")
    print(f"  (expected outputs shown above in comments)")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="IndicLID Pipeline — Identify language of any Indian text"
    )
    parser.add_argument(
        "--text", "-t", type=str, default=None,
        help="Text to identify"
    )
    parser.add_argument(
        "--no_bert", action="store_true",
        help="Use FTN/FTR only (no BERT). Faster but less accurate."
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"FTR confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print pipeline steps"
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  IndicLID — Ensemble Pipeline (ACL 2023, AI4Bharat)    ║")
    print("║  Figure 1: FTN / FTR + BERT ensemble                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    pipeline = IndicLIDPipeline(
        roman_threshold=ROMAN_CHAR_THRESHOLD,
        confidence_threshold=args.threshold,
        use_bert=not args.no_bert,
    )

    if args.text:
        result = pipeline.identify(args.text, verbose=args.verbose)
        print(f"\n  Input      : '{args.text}'")
        print(f"  Prediction : {result['label']}")
        print(f"  Language   : {pipeline.get_language_name(result['label'])}")
        print(f"  Confidence : {result['confidence']:.3f}")
        print(f"  Model used : {result['model_used']}")
    else:
        print("  Running demo with test sentences …\n")
        run_demo(pipeline)

    print("\n  Next step: python 07_evaluate.py")
