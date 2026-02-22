================================================================
BharatBhasaNet — Step-by-Step Implementation Guide
(IEEE Access 2024 | IIT Roorkee + NIC)
================================================================

WHAT THIS PROJECT DOES
───────────────────────
Identifies the language of every word in a code-mixed Indian
sentence. Supports 12 languages in both native script and
Romanized (Latin) form.

Example:
  Input : "Hello Aap kaise hain? स्वागत है आपका। আপনি কেমন আছেন?"
  Output: English | Hindi | Hindi | Bengali


PROJECT STRUCTURE
─────────────────
BharatBhasaNet/
│
├── config.py                   ← All settings (paths, labels, hyperparams)
│
├── 01_download_datasets.py     ← Download IndicCorp + Aksharantar + Bhasha-Abhijnaanam
├── 02_preprocess_native.py     ← Clean native-script CSVs
├── 03_preprocess_romanized.py  ← Clean Romanized CSVs
├── 04_train_native.py          ← Fine-tune XLM-RoBERTa for native scripts
├── 05_train_romanized.py       ← Fine-tune XLM-RoBERTa for Romanized text
├── 06_transliteration.py       ← Romanized → Native script converter
├── 07_pipeline.py              ← Full 2-model pipeline + Beam Search
├── 08_evaluate.py              ← Reproduce Tables 1 & 2 from the paper
├── 09_inference.py             ← Easy inference on any sentence
│
├── data/
│   ├── raw/
│   │   ├── native/             ← Raw native-script .txt files
│   │   └── romanized/          ← Raw Romanized .txt files
│   └── processed/              ← Cleaned training/test CSVs
│
├── models/
│   ├── roberta_native/         ← Saved Native model checkpoint
│   └── roberta_romanized/      ← Saved Romanized model checkpoint
│
└── logs/                       ← Training logs


HOW TO RUN — COMPLETE SEQUENCE
───────────────────────────────

STEP 0 — Install dependencies
  cd BharatBhasaNet
  pip install -r requirements.txt

STEP 1 — Download all 3 datasets  (~8-15 GB)
  python 01_download_datasets.py
  → Downloads: IndicCorp (native) + Aksharantar + Bhasha-Abhijnaanam
  → Saves to: data/raw/

STEP 2 — Preprocess native data
  python 02_preprocess_native.py
  → Cleans sentences (removes numbers, punctuation, Latin chars)
  → Saves: data/processed/native_train.csv  (600k rows)
            data/processed/native_test.csv   (360k rows)

STEP 3 — Preprocess Romanized data
  python 03_preprocess_romanized.py
  → Combines Aksharantar + Bhasha-Abhijnaanam
  → Saves: data/processed/romanized_train.csv
            data/processed/romanized_test.csv

STEP 4 — Train Native model  (~6-12 hours on RTX 3090)
  python 04_train_native.py
  → Base model : xlm-roberta-base (HuggingFace)
  → Epochs     : 10
  → LR         : 2e-5
  → Target acc : 99.54%
  → Saves to   : models/roberta_native/

STEP 5 — Train Romanized model  (~3-6 hours)
  python 05_train_romanized.py
  → Same architecture, different data
  → Target acc : 60.90%
  → Saves to   : models/roberta_romanized/

STEP 6 — Test transliteration module
  python 06_transliteration.py
  → Tests: "kaise" → "कैसे", "ami" → "আমি", etc.
  → Uses AI4Bharat IndicXlit

STEP 7 — Run the full pipeline
  python 07_pipeline.py
  → Demo with example sentences from the paper
  → Or: python 07_pipeline.py --sentence "Your text here"

STEP 8 — Evaluate (reproduce Tables 1 & 2)
  python 08_evaluate.py --model both
  → Generates confusion matrices
  → Compares XLM-RoBERTa vs SVM

STEP 9 — Inference on any sentence
  python 09_inference.py
  python 09_inference.py --sentence "Kal office mein meeting hai"
  python 09_inference.py --file my_sentences.txt


EXPECTED RESULTS (from the paper)
───────────────────────────────────
┌─────────────────────────────────┬──────┬──────┬──────┬──────┐
│ Model                           │  P   │  R   │  F1  │ Acc  │
├─────────────────────────────────┼──────┼──────┼──────┼──────┤
│ SVM Native + Count Vectorizer   │96.93 │96.55 │96.65 │96.54 │
│ SVM Native + TF-IDF Vectorizer  │97.41 │97.35 │97.35 │97.35 │
│ XLM-RoBERTa Native              │99.55 │99.54 │99.54 │99.54 │ ←best
├─────────────────────────────────┼──────┼──────┼──────┼──────┤
│ SVM Romanized                   │42.96 │21.82 │17.78 │21.82 │
│ XLM-RoBERTa Romanized           │63.90 │60.90 │61.31 │60.90 │ ←best
└─────────────────────────────────┴──────┴──────┴──────┴──────┘

Real-time NIC dataset pipeline accuracy: 92.76%


LANGUAGE LABELS
───────────────
  0 = English      1 = Bengali     2 = Assamese
  3 = Hindi        4 = Marathi     5 = Tamil
  6 = Telugu       7 = Kannada     8 = Malayalam
  9 = Gujarati    10 = Oriya      11 = Urdu
 12 = Punjabi


HARDWARE USED IN PAPER
───────────────────────
  CPU  : Intel i7 12th Gen @ 2.70 GHz
  RAM  : 128 GB
  GPU  : NVIDIA RTX 3090 Ti (24 GB VRAM)

If your GPU has less memory, reduce per_device_train_batch_size
in config.py and increase gradient_accumulation_steps accordingly.
Effective batch size should stay ~1280 to match the paper.


PAPER REFERENCE
───────────────
Sayantan Dey, Shivam Thakur, Akhilesh Kandwal, Rohit Kumar,
Sharmistha Dasgupta, Partha Pratim Roy
"BharatBhasaNet — A Unified Framework to Identify Indian Code
Mix Languages"
IEEE Access, 2024. DOI: 10.1109/ACCESS.2024.3396290
================================================================
