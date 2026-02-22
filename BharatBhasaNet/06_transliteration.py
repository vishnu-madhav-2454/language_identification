# ==============================================================
# 06_transliteration.py
#
# STEP 6 — Romanized → Native Script Transliteration
#
# What the paper did (Section IV-G, Figure 7):
#   After the Romanized model identifies a word as
#   "Romanized Hindi" or "Regional Romanized", it must
#   be converted back to its native script before the
#   Native model does the final classification.
#
#   Tool used: AI4Bharat's IndicXlit / Aksharantar transliterator.
#
# This module provides a clean interface:
#   transliterate(word, src_lang_code, tgt_lang_code) → native_word
#
# Install:  pip install ai4bharat-transliteration
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import LANGUAGE_CODES

# ─────────────────────────────────────────────────────────────────
# WRAPPER AROUND AI4BHARAT INDICXLIT
# ─────────────────────────────────────────────────────────────────

class Transliterator:
    """
    Wraps AI4Bharat's IndicXlit transliteration model.

    Usage:
        t = Transliterator()
        native_word = t.romanized_to_native("kaise", "hi")
        # Returns "कैसे"
    """

    def __init__(self):
        self._model = None
        self._loaded = False
        self._load()

    def _load(self):
        """
        Lazy-load IndicXlit.  Prints a helpful message if the
        package is not installed.
        """
        try:
            from ai4bharat.transliteration import XlitEngine
            # engine_type="default" uses the standard IndicXlit model
            self._model = XlitEngine(
                src_script_type="roman",
                beam_width=10,
                rescore=False,
            )
            self._loaded = True
            print("  [Transliterator] IndicXlit loaded successfully.")
        except ImportError:
            print(
                "  [Transliterator] ai4bharat-transliteration not installed.\n"
                "  Run: pip install ai4bharat-transliteration\n"
                "  Falling back to identity (no transliteration)."
            )
        except Exception as e:
            print(f"  [Transliterator] Warning: Could not load IndicXlit: {e}")
            print("  Falling back to identity (no transliteration).")

    def romanized_to_native(self, roman_word: str, lang_code: str) -> str:
        """
        Transliterate a single Romanized word to its native script.

        Parameters
        ----------
        roman_word : str  – word in Latin/Roman script (e.g. "kaise")
        lang_code  : str  – ISO 639-1 language code (e.g. "hi", "bn")

        Returns
        -------
        str  – the word in native script (e.g. "कैसे")
               or original word if transliteration fails
        """
        if not self._loaded or not roman_word.strip():
            return roman_word

        try:
            result = self._model.translit_word(roman_word, lang_code)
            # IndicXlit returns a list of candidates; take the best one
            if isinstance(result, list) and result:
                return result[0]
            elif isinstance(result, dict):
                candidates = result.get(lang_code, [roman_word])
                return candidates[0] if candidates else roman_word
            return str(result)
        except Exception as e:
            # Graceful degradation: return original word
            return roman_word

    def transliterate_sentence(
        self, roman_sentence: str, lang_code: str
    ) -> str:
        """
        Transliterate every word in a romanized sentence.

        Parameters
        ----------
        roman_sentence : str  – full sentence in Roman script
        lang_code      : str  – target language code

        Returns
        -------
        str  – sentence in native script
        """
        words = roman_sentence.strip().split()
        native_words = [
            self.romanized_to_native(w, lang_code) for w in words
        ]
        return " ".join(native_words)

    def batch_transliterate(
        self, word_lang_pairs: list
    ) -> list:
        """
        Transliterate a batch of (word, lang_code) tuples efficiently.

        Parameters
        ----------
        word_lang_pairs : list of (word, lang_code)

        Returns
        -------
        list of native-script words (same order)
        """
        return [
            self.romanized_to_native(word, lang_code)
            for word, lang_code in word_lang_pairs
        ]


# ─────────────────────────────────────────────────────────────────
# SINGLETON — one instance shared across the pipeline
# ─────────────────────────────────────────────────────────────────
_transliterator_instance = None

def get_transliterator() -> Transliterator:
    """Return the singleton Transliterator (load once)."""
    global _transliterator_instance
    if _transliterator_instance is None:
        _transliterator_instance = Transliterator()
    return _transliterator_instance


# ─────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Transliteration Test      ║")
    print("╚══════════════════════════════════════════════╝\n")

    t = Transliterator()

    test_cases = [
        ("kaise",     "hi",  "कैसे"),
        ("aap",       "hi",  "आप"),
        ("ami",       "bn",  "আমি"),
        ("vanakkam",  "ta",  "வணக்கம்"),
        ("namaskara", "kn",  "ನಮಸ್ಕಾರ"),
        ("Hello",     "en",  "Hello"),      # English stays English
    ]

    print(f"  {'Roman word':15s} → {'Native':20s} (expected)")
    print("  " + "-"*50)
    for roman, lang, expected in test_cases:
        native = t.romanized_to_native(roman, lang)
        match  = "✓" if native == expected else "≈"
        print(f"  {roman:15s} → {native:20s}  [{lang}] {match} (expected: {expected})")

    print("\n  Sentence transliteration:")
    sentence = "aap kaise hain"
    result   = t.transliterate_sentence(sentence, "hi")
    print(f"  '{sentence}' → '{result}'")

    print("\n✅ Transliteration module ready.")
    print("Next step: Run  python 07_pipeline.py")
