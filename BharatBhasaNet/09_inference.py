# ==============================================================
# 09_inference.py
#
# STEP 9 — Easy Inference Script
#
# This is the simplest entry point.
# Just give it a sentence and it identifies the language
# of every word using the full BharatBhasaNet pipeline.
#
# Run:
#   python 09_inference.py
#   python 09_inference.py --sentence "Hello Aap kaise hain?"
#   python 09_inference.py --file my_sentences.txt
# ==============================================================

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import LANGUAGE_LABELS


def print_result(sentence: str, result: dict):
    """Pretty-print the identification result for one sentence."""
    print(f"\n  Input   : {sentence}")
    print(f"  Output  : {result['sequence']}")
    print(f"  {'Word':25s}  Language")
    print(f"  {'─'*40}")
    for word, lang in zip(result["words"], result["language_names"]):
        print(f"  {word:25s}  {lang}")


def load_pipeline():
    """Load the full pipeline (models loaded once, reused for all sentences)."""
    # Delay heavy imports until needed
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("pipeline", Path(__file__).parent / "07_pipeline.py")
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    pipeline = mod.BharatBhasaNetPipeline()
    return pipeline


def run_on_sentences(sentences: list, pipeline=None, verbose: bool = False):
    """Run the pipeline on a list of sentences and print results."""
    if pipeline is None:
        print("[Loading models …]")
        pipeline = load_pipeline()
        print("[Models loaded]\n")

    results = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        result = pipeline.identify(sent, verbose=verbose)
        print_result(sent, result)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="BharatBhasaNet: Indian Code-Mix Language Identifier"
    )
    parser.add_argument(
        "--sentence", "-s",
        type=str, default=None,
        help="Single sentence to identify.",
    )
    parser.add_argument(
        "--file", "-f",
        type=str, default=None,
        help="Path to a .txt file with one sentence per line.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show intermediate pipeline steps.",
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  BharatBhasaNet — Language Identification           ║")
    print("║  12 Indian Languages + English (Native + Romanized) ║")
    print("╚══════════════════════════════════════════════════════╝")

    sentences = []

    if args.sentence:
        sentences = [args.sentence]

    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            sentences = f.readlines()

    else:
        # Interactive mode with demo sentences
        print("\nNo input provided. Running demo sentences from the paper:\n")
        sentences = [
            # Code-mixed: English + Romanized Hindi + Native Hindi + Native Bengali
            "Hello everyone! Aap kaise hain? स्वागत है आपका। আপনি কেমন আছেন?",
            # Native + English mix
            "Kal meeting hai at 3 PM office mein",
            # All native Hindi
            "आज का मौसम बहुत अच्छा है",
            # All English
            "The quick brown fox jumps over the lazy dog",
            # Romanized Bengali
            "Ami bhalo achi tumi kemon acho",
            # Romanized Tamil
            "Vanakkam naan nalama irukkiren",
        ]

    if sentences:
        print("[Loading models from saved checkpoints …]")
        pipeline = load_pipeline()
        print("[Models loaded]\n")
        run_on_sentences(sentences, pipeline=pipeline, verbose=args.verbose)

    else:
        # True interactive mode
        print("[Loading models …]")
        pipeline = load_pipeline()
        print("[Models loaded]\n")
        print("Type a sentence and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                sentence = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if sentence.lower() in ("quit", "exit", "q"):
                break
            if not sentence:
                continue
            result = pipeline.identify(sentence, verbose=args.verbose)
            print_result(sentence, result)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
