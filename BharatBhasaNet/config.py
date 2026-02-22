# ==============================================================
# config.py
# Central configuration for the entire BharatBhasaNet project.
# Every path, hyperparameter, and label mapping lives here.
# ==============================================================

import os

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR            = os.path.join(BASE_DIR, "data")
RAW_NATIVE_DIR      = os.path.join(DATA_DIR, "raw", "native")
RAW_ROMANIZED_DIR   = os.path.join(DATA_DIR, "raw", "romanized")
PROCESSED_DIR       = os.path.join(DATA_DIR, "processed")

MODELS_DIR              = os.path.join(BASE_DIR, "models")
NATIVE_MODEL_DIR        = os.path.join(MODELS_DIR, "roberta_native")
ROMANIZED_MODEL_DIR     = os.path.join(MODELS_DIR, "roberta_romanized")

LOGS_DIR    = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────────
# LANGUAGE LABEL MAPPING  (same as in paper)
# Labels 1-12 for Indian languages + 0 for English
# ─────────────────────────────────────────────
LANGUAGE_LABELS = {
    0:  "English",
    1:  "Bengali",
    2:  "Assamese",
    3:  "Hindi",
    4:  "Marathi",
    5:  "Tamil",
    6:  "Telugu",
    7:  "Kannada",
    8:  "Malayalam",
    9:  "Gujarati",
    10: "Oriya",
    11: "Urdu",
    12: "Punjabi",
}

LABEL_TO_ID = {v: k for k, v in LANGUAGE_LABELS.items()}

ID_TO_LANGUAGE = LANGUAGE_LABELS   # alias for readability

# Short codes used by AI4Bharat Aksharantar dataset
LANGUAGE_CODES = {
    "English":   "en",
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
}

CODE_TO_LABEL = {v: k for k, v in LANGUAGE_CODES.items()}

# ─────────────────────────────────────────────
# ROMANIZED MODEL CLASSES
# The Romanized model distinguishes these 3 super-classes
# before assigning a specific Indian language label
# True English / Romanized Hindi / Regional Romanized
# ─────────────────────────────────────────────
ROMANIZED_SUPER_CLASSES = {
    "true_english":       0,   # genuinely English
    "romanized_hindi":    3,   # Hindi written in Latin
    "regional_romanized": -1,  # any other Indian language in Latin
}

# ─────────────────────────────────────────────
# BASE MODEL (XLM-RoBERTa from HuggingFace)
# ─────────────────────────────────────────────
BASE_MODEL_NAME = "xlm-roberta-base"

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS  (from the paper)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# GPU MEMORY NOTE
# Paper used NVIDIA RTX 3090 Ti (24 GB VRAM).
# This config is tuned for RTX 3050 Laptop (4 GB VRAM):
#   per_device_train_batch_size = 4
#   gradient_accumulation_steps = 320
#   effective batch = 4 × 320 = 1280  ← same as paper
# dataloader_num_workers = 0  (required on Windows)
# ─────────────────────────────────────────────

NATIVE_TRAINING_ARGS = {
    "num_train_epochs":               10,
    "learning_rate":                  2e-5,
    "per_device_train_batch_size":    4,     # 4 GB VRAM safe
    "per_device_eval_batch_size":     8,
    "gradient_accumulation_steps":    320,   # 4×320 = 1280 effective (paper)
    "warmup_steps":                   500,
    "weight_decay":                   0.01,
    "logging_dir":                    LOGS_DIR,
    "evaluation_strategy":            "epoch",
    "save_strategy":                  "epoch",
    "load_best_model_at_end":         True,
    "metric_for_best_model":          "accuracy",
    "fp16":                           True,
    "dataloader_num_workers":         0,     # 0 required on Windows
    "output_dir":                     NATIVE_MODEL_DIR,
}

ROMANIZED_TRAINING_ARGS = {
    "num_train_epochs":               10,
    "learning_rate":                  2e-5,
    "per_device_train_batch_size":    4,
    "per_device_eval_batch_size":     8,
    "gradient_accumulation_steps":    320,
    "warmup_steps":                   200,
    "weight_decay":                   0.01,
    "logging_dir":                    LOGS_DIR,
    "evaluation_strategy":            "epoch",
    "save_strategy":                  "epoch",
    "load_best_model_at_end":         True,
    "metric_for_best_model":          "accuracy",
    "fp16":                           True,
    "dataloader_num_workers":         0,
    "output_dir":                     ROMANIZED_MODEL_DIR,
}

# ─────────────────────────────────────────────
# TOKENIZER SETTINGS
# ─────────────────────────────────────────────
MAX_TOKEN_LENGTH = 128   # max subword tokens per sentence

# ─────────────────────────────────────────────
# DATASET SIZES  (from the paper, for reference)
# ─────────────────────────────────────────────
NATIVE_TRAIN_SIZE   = 600_000   # 50,000 sentences × 12 languages
NATIVE_TEST_SIZE    = 360_000   # 30,000 sentences × 12 languages

# ─────────────────────────────────────────────
# BEAM SEARCH
# ─────────────────────────────────────────────
BEAM_WIDTH = 3   # number of candidate sequences to keep at each step

# ─────────────────────────────────────────────
# PROCESSED FILE PATHS
# ─────────────────────────────────────────────
NATIVE_TRAIN_CSV    = os.path.join(PROCESSED_DIR, "native_train.csv")
NATIVE_TEST_CSV     = os.path.join(PROCESSED_DIR, "native_test.csv")
ROMANIZED_TRAIN_CSV = os.path.join(PROCESSED_DIR, "romanized_train.csv")
ROMANIZED_TEST_CSV  = os.path.join(PROCESSED_DIR, "romanized_test.csv")
