================================================================
IndicLID — Language Identification for 22 Indic Languages
Implementation of: "Bhasha-Abhijnaanam: Native-script and
Romanized Language Identification for 22 Indic Languages"
ACL 2023 (Short Papers, pp. 816–826)
Authors: Yash Madhani, Mitesh M. Khapra, Anoop Kunchukuttan
AI4Bharat / IIT Madras / Microsoft
================================================================

WHAT THIS PROJECT DOES
──────────────────────
Given any text input (sentence or word), this system identifies
which of the 22 official Indian constitutional languages it
belongs to — in both native script AND romanized (Latin) form.

Input examples:
  "नमस्ते आप कैसे हैं"     → hi_Deva   (Hindi, Devanagari)
  "namaste aap kaise hain"  → hi_Latn   (Hindi, Romanized)
  "vanakkam"                → ta_Latn   (Tamil, Romanized)
  "வணக்கம்"                  → ta_Taml   (Tamil, Tamil script)

ARCHITECTURE (from the paper)
──────────────────────────────
Three model variants:

  1. IndicLID-FTN  (FastText, 8-dim, native script)
     • Character n-gram features
     • Fast: ~30,000 sentences/second
     • 98% accuracy on native script test set

  2. IndicLID-FTR  (FastText, 8-dim, romanized)
     • Same architecture, romanized training data
     • ~71% accuracy on romanized test set

  3. IndicLID-BERT (IndicBERT fine-tuned, 1 layer unfrozen)
     • Slower but more accurate for romanized text
     • ~80% accuracy on romanized test set

  4. IndicLID (Ensemble Pipeline):
     ┌─────────────────────────────────────────────────────┐
     │  Is >50% of input in Roman characters?              │
     │    YES → IndicLID-FTR (fast linear classifier)      │
     │          ├─ Confidence > 0.6 → Return FTR result    │
     │          └─ Confidence ≤ 0.6 → IndicLID-BERT result │
     │    NO  → IndicLID-FTN (native script classifier)    │
     └─────────────────────────────────────────────────────┘

  Ensemble throughput: ~10 sentences/second (3x faster than BERT-only)

47 OUTPUT CLASSES
──────────────────
  Native (24 classes):  as_Beng, bn_Beng, brx_Deva, doi_Deva,
    gu_Gujr, hi_Deva, kn_Knda, ks_Arab, ks_Deva, kok_Deva,
    mai_Deva, ml_Mlym, mni_Beng, mni_Mtei, mr_Deva, ne_Deva,
    or_Orya, pa_Guru, sa_Deva, sat_Olck, sd_Arab, ta_Taml,
    te_Telu, ur_Arab

  Romanized (21 classes):  as_Latn, bn_Latn, brx_Latn, doi_Latn,
    gu_Latn, hi_Latn, kn_Latn, ks_Latn, kok_Latn, mai_Latn,
    ml_Latn, mni_Latn, mr_Latn, ne_Latn, or_Latn, pa_Latn,
    sa_Latn, sd_Latn, ta_Latn, te_Latn, ur_Latn

  Special (2 classes):  en (English), xx (Others)

NOTE: Santali (sat) has no romanized class. Dogri (doi) romanized
support is limited since IndicXlit does not support Dogri.

DATASETS
─────────
  Native training data:
    • IndicCorpV2 (ai4bharat/IndicCorpV2)
    • 100,000 sentences per language (22 languages + English)
    • Sources: IndicCorp, NLLB, Wikipedia, Vikaspedia

  Romanized training data (SYNTHETIC):
    • Created by transliterating native training data via IndicXlit
    • (IndicXlit: ai4bharat/IndicXlit / ai4bharat-transliteration)
    • Real romanized corpora are scarce for Indian languages

  Benchmark / Test set:
    • Bhasha-Abhijnaanam (ai4bharat/Bhasha-Abhijnaanam)
    • Native: built from FLORES-200 + Dakshina + new translations
    • Romanized: Dakshina (filtered) + IndicCorp + new annotations

PROJECT STRUCTURE
──────────────────
IndicLID/
│
├── config.py                        ← All settings, label maps, paths
├── requirements.txt
├── README.txt
│
├── 01_download_datasets.py          ← Download IndicCorpV2 + Bhasha-Abhijnaanam
├── 02_preprocess_native.py          ← Clean native data → FastText format
├── 03_generate_synthetic_romanized.py ← Transliterate native → romanized
├── 04_train_fasttext.py             ← Train IndicLID-FTN & IndicLID-FTR
├── 05_train_indicbert.py            ← Fine-tune IndicBERT (romanized)
├── 06_pipeline.py                   ← Ensemble: FTN + FTR + BERT
├── 07_evaluate.py                   ← Evaluate on Bhasha-Abhijnaanam benchmark
├── 08_inference.py                  ← Easy inference on any text
│
├── data/
│   ├── raw/
│   │   ├── native/                  ← Raw native .txt files per language
│   │   └── romanized/               ← IndicCorpV2 romanized (if available)
│   ├── synthetic_romanized/         ← Transliterated romanized training data
│   └── processed/                   ← FastText .txt and .csv files
│
├── models/
│   ├── indiclid_ftn.bin             ← FastText native model
│   ├── indiclid_ftr.bin             ← FastText romanized model
│   └── indiclid_bert/               ← IndicBERT fine-tuned checkpoint
│
└── logs/


HOW TO RUN — COMPLETE SEQUENCE
────────────────────────────────

STEP 0 — Install dependencies
  pip install -r requirements.txt

STEP 1 — Download datasets  (~5–15 GB total)
  python 01_download_datasets.py
  → Downloads IndicCorpV2 (native sentences per language)
  → Downloads Bhasha-Abhijnaanam benchmark test set
  → Saves raw data to: data/raw/

STEP 2 — Preprocess native data
  python 02_preprocess_native.py
  → Cleans native-script sentences
  → Removes digits, punctuation, cross-script contamination
  → Generates FastText training format files
  → Saves: data/processed/native_train_ft.txt
            data/processed/native_test_ft.txt
            data/processed/native_train.csv
            data/processed/native_test.csv

STEP 3 — Generate synthetic romanized data
  python 03_generate_synthetic_romanized.py
  → Transliterates native training data → romanized
  → Uses AI4Bharat IndicXlit (must be installed)
  → This is CRITICAL: paper shows synthetic data enables romanized LID
  → Saves: data/synthetic_romanized/<lang>/train.txt
            data/processed/romanized_train_ft.txt
            data/processed/romanized_train.csv

STEP 4 — Train FastText models
  python 04_train_fasttext.py
  → Trains IndicLID-FTN  (native script, 8-dim)
  → Trains IndicLID-FTR  (romanized, 8-dim)
  → Expected: FTN ~98% acc, FTR ~71% acc
  → Saves: models/indiclid_ftn.bin
            models/indiclid_ftr.bin

STEP 5 — Fine-tune IndicBERT for romanized text
  python 05_train_indicbert.py
  → Base: ai4bharat/IndicBERTv2-MLM-only
  → Freezes all layers except last 1 transformer layer
  → Trains on synthetic romanized data
  → Expected: ~80% accuracy on romanized test set
  → Saves: models/indiclid_bert/

STEP 6 — Test the full pipeline (no training needed)
  python 06_pipeline.py
  → Demonstrates the ensemble pipeline
  → python 06_pipeline.py --text "your sentence here"

STEP 7 — Evaluate on Bhasha-Abhijnaanam benchmark
  python 07_evaluate.py --model all
  → Evaluates FTN, FTR, BERT, Ensemble on benchmark
  → Generates confusion matrices
  → Reports match paper: FTN ~98%, Ensemble ~80%

STEP 8 — Easy inference
  python 08_inference.py
  → python 08_inference.py --text "kaise ho aap"
  → python 08_inference.py --interactive
  → python 08_inference.py --file sentences.txt

KEY RESULTS FROM THE PAPER (Table 3 & 4)
──────────────────────────────────────────
  Native script (IndicLID-FTN):
    Accuracy : 98.55%
    Precision: 98.11%   Recall: 98.56%   F1: 98.31%
    Throughput: 30,303 sentences/second
    Model size: 318 MB

  Romanized (IndicLID Ensemble, threshold=0.6):
    Accuracy : 80.40%
    Precision: 72.74%   Recall: 84.50%   F1: 74.72%
    Throughput: ~10 sentences/second
    Model size: ~1.4 GB

  Comparison (native) vs other LIDs:
    IndicLID-FTN : 98.55%  30,303 sent/s  318M
    CLD3         : 98.03%   4,861 sent/s   —
    NLLB         : 98.78%   4,970 sent/s  1.1G
    (IndicLID is 6-10x faster than NLLB with comparable accuracy)

REFERENCES
───────────
  Paper: https://aclanthology.org/2023.acl-short.71
  GitHub: https://github.com/AI4Bharat/IndicLID
  Data: https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam
  IndicXlit: https://github.com/AI4Bharat/IndicXlit
  IndicBERT: https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only
