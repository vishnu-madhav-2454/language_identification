# ==============================================================
# 08_inference.py
#
# STEP 8 — IndicLID Inference (End-User Entry Point)
#
# Easy, ready-to-use inference for any Indic text.
# Wraps the 06_pipeline.py IndicLIDPipeline.
#
# Usage (single text):
#   python 08_inference.py --text "namaste aap kaise hain"
#   python 08_inference.py --text "নমস্কার আপনি কেমন আছেন"
#
# Usage (file):
#   python 08_inference.py --file sentences.txt
#   python 08_inference.py --file sentences.txt --output results.csv
#
# Usage (interactive shell):
#   python 08_inference.py --interactive
#   python 08_inference.py --interactive --no_bert
#
# Supported output formats:
#   --format text     (default — human-readable)
#   --format json     (JSON lines)
#   --format csv      (CSV, requires --output)
# ==============================================================

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ROMAN_CHAR_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    ID_TO_LABEL,
    INDIC_LANGUAGES,
)


# ─────────────────────────────────────────────────────────────────
# LOAD PIPELINE DYNAMICALLY (avoids "06_" import issue)
# ─────────────────────────────────────────────────────────────────

def load_pipeline(use_bert: bool = True, threshold: float = CONFIDENCE_THRESHOLD):
    """Load the IndicLID ensemble pipeline."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "indiclid_pipeline",
        Path(__file__).parent / "06_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pipeline = mod.IndicLIDPipeline(
        roman_threshold=ROMAN_CHAR_THRESHOLD,
        confidence_threshold=threshold,
        use_bert=use_bert,
    )
    return pipeline, mod


# ─────────────────────────────────────────────────────────────────
# OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────────

def format_result_text(text: str, result: dict, pipeline_mod) -> str:
    """Format a single result as human-readable text."""
    label      = result["label"]
    confidence = result["confidence"]
    model      = result["model_used"]
    is_roman   = result["is_roman"]

    # Build language name
    if label == "en":
        lang_name = "English"
    elif label in ("xx", "others"):
        lang_name = "Others / Unknown"
    else:
        parts  = label.split("_")
        iso    = parts[0]
        script = "Romanized" if label.endswith("_Latn") else "Native"
        info   = INDIC_LANGUAGES.get(iso)
        if info:
            lang_name = f"{info[0]} ({script})"
        else:
            lang_name = label

    script_type = "Romanized" if is_roman else "Native-script"
    return (
        f"  ┌─ Input  : {text}\n"
        f"  ├─ Label  : {label}  ({lang_name})\n"
        f"  ├─ Script : {script_type}\n"
        f"  ├─ Conf.  : {confidence:.4f}\n"
        f"  └─ Model  : {model}"
    )


def format_result_json(text: str, result: dict) -> str:
    """Format a single result as a JSON line."""
    out = {
        "text":       text,
        "label":      result["label"],
        "confidence": round(result["confidence"], 4),
        "model_used": result["model_used"],
        "is_roman":   result["is_roman"],
    }
    return json.dumps(out, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────
# FILE INFERENCE
# ─────────────────────────────────────────────────────────────────

def run_file_inference(
    input_path: str,
    pipeline,
    pipeline_mod,
    output_path: Optional[str] = None,
    fmt: str = "text",
    batch_size: int = 256,
):
    """Run inference on all lines in a text file."""
    if not os.path.exists(input_path):
        print(f"  File not found: {input_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    print(f"\n  Loaded {len(lines)} lines from: {input_path}")
    print(f"  Running inference …")

    results = []
    t0  = time.time()

    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        for text in batch:
            result = pipeline.identify(text)
            results.append((text, result))
        if (i // batch_size + 1) % 5 == 0:
            print(f"    … {i + len(batch)} / {len(lines)}", end="\r")

    elapsed    = time.time() - t0
    throughput = len(lines) / elapsed if elapsed > 0 else 0
    print(f"\n  Done. {len(lines)} samples in {elapsed:.1f}s ({throughput:.0f} sent/s)")

    # ── Write or print output ──
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        if fmt == "csv" or ext == ".csv":
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label", "confidence", "model_used", "is_roman"])
                for text, result in results:
                    writer.writerow([
                        text,
                        result["label"],
                        round(result["confidence"], 4),
                        result["model_used"],
                        result["is_roman"],
                    ])
        elif fmt == "json" or ext == ".jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for text, result in results:
                    f.write(format_result_json(text, result) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                for text, result in results:
                    f.write(format_result_text(text, result, pipeline_mod) + "\n\n")
        print(f"  Results saved to: {output_path}")
    else:
        # Print to stdout
        for text, result in results[:50]:  # limit to 50 if no output file
            if fmt == "json":
                print(format_result_json(text, result))
            else:
                print(format_result_text(text, result, pipeline_mod))
                print()
        if len(results) > 50:
            print(f"  (showing first 50 of {len(results)} — use --output to save all)")


# ─────────────────────────────────────────────────────────────────
# INTERACTIVE SHELL
# ─────────────────────────────────────────────────────────────────

INTERACTIVE_HELP = """
  Commands:
    <any text>       Identify the language of the text
    :verbose on/off  Toggle verbose mode (show pipeline steps)
    :threshold N     Set FTR confidence threshold (0.0 – 1.0)
    :help            Show this help
    :quit / :exit    Exit
"""


def run_interactive(pipeline, pipeline_mod):
    """Interactive inference shell."""
    print("\n  IndicLID Interactive Mode")
    print("  Type text to identify its language.")
    print("  Type :help for commands, :quit to exit.\n")

    verbose = False

    while True:
        try:
            text = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting …")
            break

        if not text:
            continue

        # ── Special commands ──
        if text.startswith(":"):
            cmd = text[1:].lower()
            if cmd in ("quit", "exit", "q"):
                print("  Exiting …")
                break
            elif cmd == "help":
                print(INTERACTIVE_HELP)
            elif cmd.startswith("verbose"):
                parts = cmd.split()
                verbose = len(parts) > 1 and parts[1] == "on"
                print(f"  Verbose: {'on' if verbose else 'off'}")
            elif cmd.startswith("threshold"):
                parts = cmd.split()
                if len(parts) > 1:
                    try:
                        pipeline.confidence_threshold = float(parts[1])
                        print(f"  Threshold set to: {pipeline.confidence_threshold}")
                    except ValueError:
                        print("  Invalid value. Usage: :threshold 0.6")
            else:
                print(f"  Unknown command: '{text}'. Try :help")
            continue

        # ── Inference ──
        result = pipeline.identify(text, verbose=verbose)
        print(format_result_text(text, result, pipeline_mod))
        print()


# ─────────────────────────────────────────────────────────────────
# QUICK LABEL LEGEND
# ─────────────────────────────────────────────────────────────────

def print_label_legend():
    """Print a compact legend of all 47 label classes."""
    print("\n  ── IndicLID Label Legend (47 classes) ──\n")
    print(f"  {'Label':22s}  {'Language':22s}  {'Script'}")
    print(f"  {'─'*60}")

    for label_id, label in sorted(ID_TO_LABEL.items()):
        if label == "en":
            print(f"  {'en':22s}  {'English':22s}  Latin")
            continue
        if label in ("xx", "others"):
            print(f"  {'xx':22s}  {'Others/Unknown':22s}  —")
            continue
        parts  = label.split("_")
        iso    = parts[0] if parts else label
        is_rom = label.endswith("_Latn")
        info   = INDIC_LANGUAGES.get(iso)
        if info:
            lang_name = info[0]
            script    = "Romanized (Latin)" if is_rom else info[1]
            print(f"  {label:22s}  {lang_name:22s}  {script}")
        else:
            print(f"  {label:22s}  {'':22s}  —")

    print(f"\n  Total: {len(ID_TO_LABEL)} labels")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IndicLID — Identify the language of any Indian text (ACL 2023)"
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text", "-t", type=str, default=None,
        help="Single text to identify",
    )
    input_group.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to a text file (one sentence per line)",
    )
    input_group.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start interactive shell",
    )
    input_group.add_argument(
        "--labels", action="store_true",
        help="Print the 47-class label legend and exit",
    )

    # Options
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (for --file mode)",
    )
    parser.add_argument(
        "--format", choices=["text", "json", "csv"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--no_bert", action="store_true",
        help="Use FTR only — faster but less accurate on ambiguous romanized text",
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"FTR confidence threshold for escalating to BERT (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show internal pipeline steps",
    )

    args = parser.parse_args()

    # ── Label legend ──
    if args.labels:
        print_label_legend()
        return

    # ── Banner ──
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  IndicLID — Indic Language Identifier               ║")
    print("║  ACL 2023, AI4Bharat/IIT Madras/Microsoft           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n  Mode    : {'FTN + FTR only' if args.no_bert else 'FTN + FTR + BERT ensemble'}")
    print(f"  Threshold: {args.threshold}")

    # ── Load pipeline ──
    pipeline, pipeline_mod = load_pipeline(
        use_bert=not args.no_bert,
        threshold=args.threshold,
    )

    # ── Dispatch ──
    if args.text:
        result = pipeline.identify(args.text, verbose=args.verbose)
        if args.format == "json":
            print(format_result_json(args.text, result))
        else:
            print(format_result_text(args.text, result, pipeline_mod))

    elif args.file:
        run_file_inference(
            input_path=args.file,
            pipeline=pipeline,
            pipeline_mod=pipeline_mod,
            output_path=args.output,
            fmt=args.format,
        )

    elif args.interactive:
        run_interactive(pipeline, pipeline_mod)

    else:
        # Default: run the built-in demo
        print("\n  No input provided. Running built-in demo …")
        from module_06_pipeline import run_demo  # type: ignore
        run_demo(pipeline)
        print("\n  Use --help for usage options.")
        print("  Use --labels to see all 47 language classes.")


# ── Pre-load pipeline module alias ──
import importlib.util as _ilu
def _preload_pipeline():
    spec = _ilu.spec_from_file_location(
        "module_06_pipeline",
        Path(__file__).parent / "06_pipeline.py",
    )
    mod = _ilu.module_from_spec(spec)
    import sys as _sys
    _sys.modules["module_06_pipeline"] = mod
    spec.loader.exec_module(mod)

_preload_pipeline()


if __name__ == "__main__":
    main()
