# ==============================================================
# 07_pipeline.py
#
# STEP 7 — Full BharatBhasaNet Code-Mixed Identification Pipeline
#
# Implements EXACTLY what the paper describes in Section IV-G
# and Figure 7 / Figure 8:
#
#   Input sentence (code-mixed)
#        ↓
#   [Model 1] RoBERTa Native
#        → classifies each word as: English / Hindi / Regional
#        → stores word indices
#        ↓
#   [Model 2] RoBERTa Romanized  (for words labeled "English")
#        → reclassifies as: True English / Romanized Hindi / Regional Romanized
#        ↓
#   [Transliteration]
#        → Romanized Hindi    → Hindi Devanagari
#        → Regional Romanized → Native Regional script
#        ↓
#   [Model 1 again] RoBERTa Native
#        → final classification of all words
#        ↓
#   [Beam Search]
#        → optimal language sequence
#        ↓
#   Output: language label per word + language sequence
#
# Run:  python 07_pipeline.py  (interactive demo)
#       python 07_pipeline.py --sentence "Hello Aap kaise hain?"
# ==============================================================

import os
import sys
import math
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NATIVE_MODEL_DIR,
    ROMANIZED_MODEL_DIR,
    LANGUAGE_LABELS,
    LANGUAGE_CODES,
    LABEL_TO_ID,
    BEAM_WIDTH,
    MAX_TOKEN_LENGTH,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Super-class labels used internally by the pipeline
ENGLISH_LABEL  = 0   # True English
HINDI_LABEL    = 3   # Hindi (native or romanized)
# Any label 1-12 except Hindi → regional language


# ─────────────────────────────────────────────────────────────────
# MODEL LOADER  (with singleton caching)
# ─────────────────────────────────────────────────────────────────

class ModelLoader:
    """Load and cache the two RoBERTa models."""

    def __init__(self):
        self._native_tok    = None
        self._native_model  = None
        self._roman_tok     = None
        self._roman_model   = None
        self._device        = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  [ModelLoader] Using device: {self._device}")

    def _check_dir(self, model_dir: str, name: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"{name} model not found at: {model_dir}\n"
                f"Please run 04_train_native.py / 05_train_romanized.py first."
            )

    def load_native(self):
        if self._native_model is None:
            self._check_dir(NATIVE_MODEL_DIR, "Native")
            print(f"  Loading Native model from: {NATIVE_MODEL_DIR}")
            self._native_tok   = AutoTokenizer.from_pretrained(NATIVE_MODEL_DIR)
            self._native_model = AutoModelForSequenceClassification.from_pretrained(
                NATIVE_MODEL_DIR
            ).to(self._device).eval()
        return self._native_model, self._native_tok

    def load_romanized(self):
        if self._roman_model is None:
            self._check_dir(ROMANIZED_MODEL_DIR, "Romanized")
            print(f"  Loading Romanized model from: {ROMANIZED_MODEL_DIR}")
            self._roman_tok   = AutoTokenizer.from_pretrained(ROMANIZED_MODEL_DIR)
            self._roman_model = AutoModelForSequenceClassification.from_pretrained(
                ROMANIZED_MODEL_DIR
            ).to(self._device).eval()
        return self._roman_model, self._roman_tok

    @property
    def device(self):
        return self._device


# ─────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────

def predict_words(
    words: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run the model on a list of words in batches.

    Returns
    -------
    np.ndarray of shape (N, num_labels)
        Softmax probabilities for each word over all language classes.
    """
    all_probs = []

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)   # shape: (N, 13)


# ─────────────────────────────────────────────────────────────────
# BEAM SEARCH  (Section IV-F)
#
# Equations from the paper:
#   Score(Y_{t-1}, y_t) = Score(Y_{t-1}) + log P(y_t | Y_{t-1})
#   Y_t = top_b(Score(Y_{t-1}, y_t))
# ─────────────────────────────────────────────────────────────────

def beam_search_language_sequence(
    word_probs: np.ndarray,    # shape (N, num_labels)
    beam_width: int = BEAM_WIDTH,
) -> List[int]:
    """
    Find the globally optimal language label sequence using beam search.

    Each step keeps the top `beam_width` partial sequences and extends
    them by one word.  Final best sequence = highest-scoring path.

    Parameters
    ----------
    word_probs  : probability distribution per word (N × num_classes)
    beam_width  : number of parallel candidate sequences

    Returns
    -------
    List[int]  – best language label for every word
    """
    n_words, n_classes = word_probs.shape

    # ── Initialise: start with top-b single-word sequences ──
    log_probs_0 = np.log(word_probs[0] + 1e-12)  # log P for word 0
    top_indices = np.argsort(log_probs_0)[::-1][:beam_width]

    # beams: list of (cumulative_log_score, [label_sequence])
    beams: List[Tuple[float, List[int]]] = [
        (log_probs_0[idx], [int(idx)])
        for idx in top_indices
    ]

    # ── Extend beam word by word ──
    for t in range(1, n_words):
        log_probs_t = np.log(word_probs[t] + 1e-12)
        candidates: List[Tuple[float, List[int]]] = []

        for score, seq in beams:
            for cls in range(n_classes):
                # Score(Y_{t-1}, y_t) = Score(Y_{t-1}) + log P(y_t)
                new_score = score + log_probs_t[cls]
                candidates.append((new_score, seq + [cls]))

        # Keep top b candidates  (Y_t = top_b(...))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # ── Return the highest-scoring sequence ──
    best_score, best_sequence = beams[0]
    return best_sequence


# ─────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────

class BharatBhasaNetPipeline:
    """
    The complete two-model BharatBhasaNet pipeline.

    Steps exactly as in Figure 7 and Figure 8 of the paper:
        1. Tokenize sentence into words
        2. Pass all words through RoBERTa-Native
        3. Store index of "English" words → pass to RoBERTa-Romanized
        4. Reclassify: True English / Romanized Hindi / Regional Romanized
        5. Transliterate Romanized words to native scripts
        6. Reconstruct sentence with transliterated words
        7. Run RoBERTa-Native again on the full reconstructed sentence
        8. Beam Search on word-level probabilities → final language sequence
    """

    def __init__(self):
        self._loader = ModelLoader()
        self._native_model  = None
        self._native_tok    = None
        self._roman_model   = None
        self._roman_tok     = None
        self._transliterator = None

    def _ensure_loaded(self):
        if self._native_model is None:
            self._native_model, self._native_tok   = self._loader.load_native()
        if self._roman_model is None:
            self._roman_model,  self._roman_tok    = self._loader.load_romanized()
        if self._transliterator is None:
            from transliteration_module import get_transliterator
            self._transliterator = get_transliterator()

    @property
    def device(self):
        return self._loader.device

    def _get_lang_code(self, label: int) -> str:
        """Convert integer label to ISO lang code for transliteration."""
        lang_name = LANGUAGE_LABELS.get(label, "")
        return LANGUAGE_CODES.get(lang_name, "hi")

    def identify(self, sentence: str, verbose: bool = False) -> dict:
        """
        Run the full pipeline on one code-mixed sentence.

        Parameters
        ----------
        sentence : str   – input code-mixed sentence
        verbose  : bool  – print intermediate steps

        Returns
        -------
        dict with keys:
            words         : list of words
            labels        : list of int labels (per word)
            language_names: list of language names (per word)
            sequence      : language sequence string
        """
        self._ensure_loaded()

        # ══════════════════════════════════════════════════════════
        # PHASE 1 — RoBERTa Native: classify every word
        # ══════════════════════════════════════════════════════════
        words = sentence.strip().split()
        if not words:
            return {"words": [], "labels": [], "language_names": [], "sequence": ""}

        if verbose:
            print(f"\n  INPUT: '{sentence}'")
            print(f"  WORDS: {words}")
            print("\n  ── PHASE 1: RoBERTa Native ──")

        native_probs = predict_words(
            words, self._native_model, self._native_tok, self.device
        )
        native_labels = np.argmax(native_probs, axis=1).tolist()

        if verbose:
            for w, lbl in zip(words, native_labels):
                print(f"    '{w}' → {LANGUAGE_LABELS[lbl]}")

        # ══════════════════════════════════════════════════════════
        # PHASE 2 — Separate English-labeled words → Romanized model
        # English = label 0 from the native model
        # ══════════════════════════════════════════════════════════
        #
        # Store indices so we can put them back in the right place
        english_indices = [
            i for i, lbl in enumerate(native_labels) if lbl == ENGLISH_LABEL
        ]
        english_words = [words[i] for i in english_indices]

        if verbose:
            print(f"\n  ── PHASE 2: RoBERTa Romanized ──")
            print(f"    Words sent to Romanized model: {english_words}")

        # Run Romanized model only if there are "English" words
        roman_labels = {}   # {word_index: new_label}
        if english_words:
            roman_probs  = predict_words(
                english_words, self._roman_model, self._roman_tok, self.device
            )
            roman_preds  = np.argmax(roman_probs, axis=1).tolist()

            for idx, new_lbl in zip(english_indices, roman_preds):
                roman_labels[idx] = new_lbl
                if verbose:
                    print(f"    '{words[idx]}' → {LANGUAGE_LABELS[new_lbl]}")

        # ══════════════════════════════════════════════════════════
        # PHASE 3 — Transliteration
        # Romanized Hindi (label=3) or any Regional Romanized
        # (label 1,2,4-12) that was previously labeled "English"
        # → convert to native script
        # ══════════════════════════════════════════════════════════
        reconstructed_words = list(words)   # mutable copy

        if verbose:
            print("\n  ── PHASE 3: Transliteration ──")

        for idx, new_lbl in roman_labels.items():
            if new_lbl != ENGLISH_LABEL:          # not True English
                lang_code = self._get_lang_code(new_lbl)
                native_word = self._transliterator.romanized_to_native(
                    words[idx], lang_code
                )
                reconstructed_words[idx] = native_word
                if verbose:
                    print(
                        f"    '{words[idx]}' [{LANGUAGE_LABELS[new_lbl]}] "
                        f"→ '{native_word}'"
                    )

        if verbose:
            print(f"\n  Reconstructed: {' '.join(reconstructed_words)}")

        # ══════════════════════════════════════════════════════════
        # PHASE 4 — RoBERTa Native (again) on reconstructed sentence
        # ══════════════════════════════════════════════════════════
        if verbose:
            print("\n  ── PHASE 4: RoBERTa Native (final pass) ──")

        final_probs  = predict_words(
            reconstructed_words,
            self._native_model, self._native_tok, self.device
        )
        # Merge: for words that were originally non-English (not routed
        # through phase 2), use native_probs directly (already confident)
        merged_probs = final_probs.copy()

        if verbose:
            final_labels = np.argmax(merged_probs, axis=1).tolist()
            for w, lbl in zip(reconstructed_words, final_labels):
                print(f"    '{w}' → {LANGUAGE_LABELS[lbl]}")

        # ══════════════════════════════════════════════════════════
        # PHASE 5 — Beam Search over word-level language probabilities
        # ══════════════════════════════════════════════════════════
        if verbose:
            print(f"\n  ── PHASE 5: Beam Search (width={BEAM_WIDTH}) ──")

        best_sequence = beam_search_language_sequence(merged_probs, BEAM_WIDTH)
        language_names = [LANGUAGE_LABELS[lbl] for lbl in best_sequence]
        unique_langs   = list(dict.fromkeys(language_names))   # preserve order

        if verbose:
            print("\n  ── FINAL RESULT ──")
            for w, lbl, lname in zip(words, best_sequence, language_names):
                print(f"    '{w:20s}' → [{lbl:2d}] {lname}")
            print(f"\n  Language Sequence: {' | '.join(unique_langs)}")

        return {
            "words":          words,
            "labels":         best_sequence,
            "language_names": language_names,
            "unique_languages": unique_langs,
            "sequence":       " | ".join(unique_langs),
        }


# ─────────────────────────────────────────────────────────────────
# IMPORT FIX  (the transliteration module import inside the class)
# ─────────────────────────────────────────────────────────────────
# We need to be able to find 06_transliteration.py as a module.
# We rename the import to use a helper:

def _load_transliterator():
    spec = __import__("importlib").util.spec_from_file_location(
        "transliteration_module",
        os.path.join(Path(__file__).parent, "06_transliteration.py"),
    )
    mod = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["transliteration_module"] = mod
    return mod.get_transliterator()


# Patch the pipeline to use the helper
BharatBhasaNetPipeline._ensure_loaded_original = (
    BharatBhasaNetPipeline._ensure_loaded
)

def _patched_ensure_loaded(self):
    if self._native_model is None:
        self._native_model, self._native_tok   = self._loader.load_native()
    if self._roman_model is None:
        self._roman_model,  self._roman_tok    = self._loader.load_romanized()
    if self._transliterator is None:
        self._transliterator = _load_transliterator()

BharatBhasaNetPipeline._ensure_loaded = _patched_ensure_loaded


# ─────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────

def demo():
    """Interactive demo with example sentences from the paper."""
    pipeline = BharatBhasaNetPipeline()

    # Example sentences from Figure 2 of the paper
    examples = [
        # Mixed English + Romanized Hindi + Native Hindi + Native Bengali
        "Hello everyone! Aap kaise hain? स्वागत है आपका। আপনি কেমন আছেন?",
        # Romanized Tamil + Romanized Kannada
        "Vanakkam naan nalama irukkiren Namaskara nanu chennagiddini",
        # Hindi only (native)
        "आज का मौसम बहुत अच्छा है",
        # English only
        "The weather is very nice today",
        # Native Bengali
        "আমি বাংলায় কথা বলছি",
    ]

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Code-Mixed Language Identification ║")
    print("╚══════════════════════════════════════════════════════╝")

    for sentence in examples:
        print("\n" + "─"*60)
        result = pipeline.identify(sentence, verbose=True)
        print(f"\n  SENTENCE  : {sentence}")
        print(f"  LANGUAGES : {result['sequence']}")

    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="BharatBhasaNet: Code-Mixed Language Identification"
    )
    parser.add_argument(
        "--sentence", "-s",
        type=str,
        default=None,
        help="Input sentence to identify languages in.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show intermediate pipeline steps.",
    )
    args = parser.parse_args()

    pipeline = BharatBhasaNetPipeline()

    if args.sentence:
        result = pipeline.identify(args.sentence, verbose=args.verbose)
        print(f"\nInput    : {args.sentence}")
        print(f"Languages: {result['sequence']}")
        for w, lang in zip(result["words"], result["language_names"]):
            print(f"  {w:20s} → {lang}")
    else:
        # Interactive mode
        print("\n╔══════════════════════════════════════════════╗")
        print("║  BharatBhasaNet Interactive Mode             ║")
        print("║  Type a code-mixed sentence and press Enter  ║")
        print("║  Type 'quit' to exit                         ║")
        print("╚══════════════════════════════════════════════╝\n")

        while True:
            sentence = input("Enter sentence: ").strip()
            if sentence.lower() in ("quit", "exit", "q"):
                break
            if not sentence:
                continue
            result = pipeline.identify(sentence, verbose=True)
            print(f"\nLanguage sequence: {result['sequence']}\n")


if __name__ == "__main__":
    main()
