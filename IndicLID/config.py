# ==============================================================
# config.py
# Central configuration for the IndicLID / Bhasha-Abhijnaanam
# implementation (ACL 2023 — AI4Bharat).
#
# Paper: "Bhasha-Abhijnaanam: Native-script and Romanized
#         Language Identification for 22 Indic Languages"
# Authors: Yash Madhani, Mitesh M. Khapra, Anoop Kunchukuttan
# ==============================================================

import os

# ─────────────────────────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR                  = os.path.join(BASE_DIR, "data")
RAW_NATIVE_DIR            = os.path.join(DATA_DIR, "raw", "native")
RAW_ROMANIZED_DIR         = os.path.join(DATA_DIR, "raw", "romanized")
PROCESSED_DIR             = os.path.join(DATA_DIR, "processed")
SYNTHETIC_ROMANIZED_DIR   = os.path.join(DATA_DIR, "synthetic_romanized")

MODELS_DIR           = os.path.join(BASE_DIR, "models")
FTN_MODEL_PATH       = os.path.join(MODELS_DIR, "indiclid_ftn.bin")    # FastText native
FTR_MODEL_PATH       = os.path.join(MODELS_DIR, "indiclid_ftr.bin")    # FastText romanized
BERT_MODEL_DIR       = os.path.join(MODELS_DIR, "indiclid_bert")       # IndicBERT fine-tuned

LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────────────────────────────
# PROCESSED DATA PATHS
# ─────────────────────────────────────────────────────────────────
NATIVE_TRAIN_FT_TXT   = os.path.join(PROCESSED_DIR, "native_train_ft.txt")    # FastText format
NATIVE_TEST_FT_TXT    = os.path.join(PROCESSED_DIR, "native_test_ft.txt")
ROMANIZED_TRAIN_FT_TXT= os.path.join(PROCESSED_DIR, "romanized_train_ft.txt")
ROMANIZED_TEST_FT_TXT = os.path.join(PROCESSED_DIR, "romanized_test_ft.txt")

NATIVE_TRAIN_CSV      = os.path.join(PROCESSED_DIR, "native_train.csv")
NATIVE_TEST_CSV       = os.path.join(PROCESSED_DIR, "native_test.csv")
ROMANIZED_TRAIN_CSV   = os.path.join(PROCESSED_DIR, "romanized_train.csv")
ROMANIZED_TEST_CSV    = os.path.join(PROCESSED_DIR, "romanized_test.csv")

BENCHMARK_NATIVE_CSV     = os.path.join(PROCESSED_DIR, "benchmark_native.csv")
BENCHMARK_ROMANIZED_CSV  = os.path.join(PROCESSED_DIR, "benchmark_romanized.csv")

# ─────────────────────────────────────────────────────────────────
# THE 22 INDIAN CONSTITUTIONAL LANGUAGES
# (8th Schedule of the Indian Constitution)
# Each entry: iso_code → (full_name, primary_script, romanized_available)
# ─────────────────────────────────────────────────────────────────
INDIC_LANGUAGES = {
    "as":  ("Assamese",   "Bengali",     True),
    "bn":  ("Bengali",    "Bengali",     True),
    "brx": ("Bodo",       "Devanagari",  True),
    "doi": ("Dogri",      "Devanagari",  True),   # IndicXlit does NOT support Dogri
    "gu":  ("Gujarati",   "Gujarati",    True),
    "hi":  ("Hindi",      "Devanagari",  True),
    "kn":  ("Kannada",    "Kannada",     True),
    "ks":  ("Kashmiri",   "Perso-Arabic",True),
    "kok": ("Konkani",    "Devanagari",  True),
    "mai": ("Maithili",   "Devanagari",  True),
    "ml":  ("Malayalam",  "Malayalam",   True),
    "mni": ("Manipuri",   "Bengali",     True),
    "mr":  ("Marathi",    "Devanagari",  True),
    "ne":  ("Nepali",     "Devanagari",  True),
    "or":  ("Oriya",      "Oriya",       True),
    "pa":  ("Punjabi",    "Gurmukhi",    True),
    "sa":  ("Sanskrit",   "Devanagari",  True),
    "sat": ("Santali",    "Ol Chiki",    False),  # no romanized
    "sd":  ("Sindhi",     "Perso-Arabic",True),
    "ta":  ("Tamil",      "Tamil",       True),
    "te":  ("Telugu",     "Telugu",      True),
    "ur":  ("Urdu",       "Perso-Arabic",True),
}

# Kashmiri also appears in Devanagari (separate class)
# Manipuri also appears in Meetei Mayek (separate class)
DUAL_SCRIPT_LANGUAGES = {
    "ks_Deva": ("Kashmiri", "Devanagari"),
    "mni_Mtei": ("Manipuri", "Meetei Mayek"),
}

# Languages supported by IndicXlit transliterator
# Dogri is NOT supported → paper acknowledges this limitation
INDICXLIT_SUPPORTED = {
    "as", "bn", "brx", "gu", "hi", "kn", "ks", "kok",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa",
    "sd", "ta", "te", "ur",
}

# ─────────────────────────────────────────────────────────────────
# 47 CLASS LABEL SYSTEM
#
# Classes 0–23  : Native-script (24 classes)
#   0  = as_Beng    (Assamese, Bengali script)
#   1  = bn_Beng    (Bengali, Bengali script)
#   2  = brx_Deva   (Bodo, Devanagari)
#   3  = doi_Deva   (Dogri, Devanagari)
#   4  = gu_Gujr    (Gujarati)
#   5  = hi_Deva    (Hindi, Devanagari)
#   6  = kn_Knda    (Kannada)
#   7  = ks_Arab    (Kashmiri, Perso-Arabic)
#   8  = ks_Deva    (Kashmiri, Devanagari)
#   9  = kok_Deva   (Konkani, Devanagari)
#   10 = mai_Deva   (Maithili, Devanagari)
#   11 = ml_Mlym    (Malayalam)
#   12 = mni_Beng   (Manipuri, Bengali script)
#   13 = mni_Mtei   (Manipuri, Meetei Mayek)
#   14 = mr_Deva    (Marathi, Devanagari)
#   15 = ne_Deva    (Nepali, Devanagari)
#   16 = or_Orya    (Oriya)
#   17 = pa_Guru    (Punjabi, Gurmukhi)
#   18 = sa_Deva    (Sanskrit, Devanagari)
#   19 = sat_Olck   (Santali, Ol Chiki)
#   20 = sd_Arab    (Sindhi, Perso-Arabic)
#   21 = ta_Taml    (Tamil)
#   22 = te_Telu    (Telugu)
#   23 = ur_Arab    (Urdu, Perso-Arabic)
#
# Classes 24–44 : Romanized (21 classes, Santali excluded)
#   24 = as_Latn    (Assamese romanized)
#   25 = bn_Latn
#   26 = brx_Latn
#   27 = doi_Latn
#   28 = gu_Latn
#   29 = hi_Latn
#   30 = kn_Latn
#   31 = ks_Latn
#   32 = kok_Latn
#   33 = mai_Latn
#   34 = ml_Latn
#   35 = mni_Latn
#   36 = mr_Latn
#   37 = ne_Latn
#   38 = or_Latn
#   39 = pa_Latn
#   40 = sa_Latn
#   41 = sd_Latn
#   42 = ta_Latn
#   43 = te_Latn
#   44 = ur_Latn
#
# Class 45: English (en)
# Class 46: Others (xx)
# ─────────────────────────────────────────────────────────────────
NUM_LABELS = 47

# Label string → integer ID
LABEL_TO_ID = {
    # Native script
    "as_Beng": 0,  "bn_Beng": 1,  "brx_Deva": 2, "doi_Deva": 3,
    "gu_Gujr": 4,  "hi_Deva": 5,  "kn_Knda": 6,  "ks_Arab": 7,
    "ks_Deva": 8,  "kok_Deva": 9, "mai_Deva": 10, "ml_Mlym": 11,
    "mni_Beng": 12,"mni_Mtei": 13,"mr_Deva": 14,  "ne_Deva": 15,
    "or_Orya": 16, "pa_Guru": 17, "sa_Deva": 18,  "sat_Olck": 19,
    "sd_Arab": 20, "ta_Taml": 21, "te_Telu": 22,  "ur_Arab": 23,
    # Romanized
    "as_Latn": 24, "bn_Latn": 25, "brx_Latn": 26, "doi_Latn": 27,
    "gu_Latn": 28, "hi_Latn": 29, "kn_Latn": 30,  "ks_Latn": 31,
    "kok_Latn": 32,"mai_Latn": 33,"ml_Latn": 34,  "mni_Latn": 35,
    "mr_Latn": 36, "ne_Latn": 37, "or_Latn": 38,  "pa_Latn": 39,
    "sa_Latn": 40, "sd_Latn": 41, "ta_Latn": 42,  "te_Latn": 43,
    "ur_Latn": 44,
    # Special
    "en": 45, "xx": 46,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Which label IDs are "romanized"? (used for >50% roman check in pipeline)
ROMANIZED_LABEL_IDS = set(range(24, 45))    # 24–44
NATIVE_LABEL_IDS    = set(range(0, 24))     # 0–23
ENGLISH_LABEL_ID    = 45
OTHERS_LABEL_ID     = 46

# Mapping: iso_code → native label id (primary script)
ISO_TO_NATIVE_LABEL = {
    "as": 0, "bn": 1, "brx": 2, "doi": 3, "gu": 4,
    "hi": 5, "kn": 6, "ks": 7,  "kok": 9, "mai": 10,
    "ml": 11,"mni": 12,"mr": 14, "ne": 15, "or": 16,
    "pa": 17,"sa": 18, "sat": 19,"sd": 20, "ta": 21,
    "te": 22,"ur": 23, "en": 45,
}

# Mapping: iso_code → romanized label id
ISO_TO_ROMAN_LABEL = {
    "as": 24, "bn": 25, "brx": 26, "doi": 27, "gu": 28,
    "hi": 29, "kn": 30, "ks": 31,  "kok": 32, "mai": 33,
    "ml": 34, "mni": 35,"mr": 36,  "ne": 37,  "or": 38,
    "pa": 39, "sa": 40, "sd": 41,  "ta": 42,  "te": 43,
    "ur": 44,
}

# ─────────────────────────────────────────────────────────────────
# INDIC CORP V2 — file codes for IndicCorpV2 HuggingFace dataset
# ─────────────────────────────────────────────────────────────────
INDICCORPV2_FILE_CODES = {
    # iso_code (used as label) → actual filename on IndicCorpV2 HuggingFace repo
    # Source: https://huggingface.co/datasets/ai4bharat/IndicCorpV2/tree/main/data
    "as":  "as",
    "bn":  "bn",
    "brx": "bd",      # Bodo: file is bd.txt (45.1 MB)
    "doi": "dg",      # Dogri: file is dg.txt (1.42 MB)
    "gu":  "gu",
    "hi":  "hi-1",    # Hindi split: hi-1.txt, hi-2.txt, hi-3.txt (26.7 GB each)
    "kn":  "kn",
    "ks":  "ks",
    "kok": "gom",     # Konkani (Goan): file is gom.txt (533 MB)
    "mai": "mai",
    "ml":  "ml",
    "mni": "mni",
    "mr":  "mr",
    "ne":  "ne",
    "or":  "or",
    "pa":  "pa",
    "sa":  "sa",
    "sat": "sat",
    "sd":  "sd",
    "ta":  "ta",
    "te":  "te",
    "ur":  "ur",
    "en":  "en",
}

# ─────────────────────────────────────────────────────────────────
# FASTTEXT HYPERPARAMETERS (from paper + Appendix A)
# Dimension=8 is the sweet spot (Table 6 in paper)
# ─────────────────────────────────────────────────────────────────
FASTTEXT_PARAMS = {
    "dim":         8,       # word vector dimension
    "epoch":       25,      # number of training epochs
    "lr":          0.5,     # learning rate
    "wordNgrams":  2,       # word n-gram size
    "minn":        2,       # min char n-gram length
    "maxn":        6,       # max char n-gram length
    "minCount":    1,       # minimum word count
    "loss":        "softmax",
    "thread":      4,
    "verbose":     2,
}

# ─────────────────────────────────────────────────────────────────
# INDICBERT HYPERPARAMETERS (from paper + Appendix B)
# Unfreeze only the last 1 transformer layer (best result)
# ─────────────────────────────────────────────────────────────────
INDICBERT_MODEL_NAME  = "ai4bharat/IndicBERTv2-MLM-only"
MAX_TOKEN_LENGTH      = 128
INDICBERT_NUM_UNFREEZE_LAYERS = 1    # paper: unfreeze-layer-1 is best

INDICBERT_TRAINING_ARGS = {
    "num_train_epochs":              10,
    "learning_rate":                 2e-5,
    "per_device_train_batch_size":   8,
    "per_device_eval_batch_size":    16,
    "gradient_accumulation_steps":   16,   # effective batch = 128
    "warmup_steps":                  500,
    "weight_decay":                  0.01,
    "eval_strategy":                  "epoch",
    "save_strategy":                 "epoch",
    "load_best_model_at_end":        True,
    "metric_for_best_model":         "accuracy",
    "fp16":                          True,
    "gradient_checkpointing":        True, # saves VRAM on 4GB GPU
    "dataloader_num_workers":        0,    # Windows requires 0
    "report_to":                     "none",
}

# ─────────────────────────────────────────────────────────────────
# PIPELINE PARAMETERS (from paper + Appendix C)
# Figure 1 / Table 9: threshold=0.6 is the best trade-off
# ─────────────────────────────────────────────────────────────────
ROMAN_CHAR_THRESHOLD   = 0.50   # >50% roman chars → use romanized model
CONFIDENCE_THRESHOLD   = 0.60   # FTR confidence < 0.6 → escalate to BERT
TRAINING_SAMPLES_PER_LANG = 100_000   # 100k per language (paper)
