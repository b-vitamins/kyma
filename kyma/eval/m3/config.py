"""Configuration constants for the optional M3 and CLaMP3 evaluators."""

from __future__ import annotations

EVAL_SPLIT = 0.01
WANDB_KEY = "<YOUR_WANDB_KEY>"

M3_TRAIN_FOLDERS = [
    "<YOUR_TRAINING_DATA_FOLDER>",
]
M3_EVAL_FOLDERS = [
    "<YOUR_EVALUATION_DATA_FOLDER>",
]

PATCH_SIZE = 64
PATCH_LENGTH = 512
PATCH_NUM_LAYERS = 12
TOKEN_NUM_LAYERS = 3
M3_HIDDEN_SIZE = 768

M3_NUM_EPOCH = 100
M3_LEARNING_RATE = 1e-4
M3_BATCH_SIZE = 16
M3_MASK_RATIO = 0.45
M3_DETERMINISTIC = True
M3_WANDB_LOG = True
M3_LOAD_CKPT = True

M3_WEIGHTS_PATH = (
    "weights_m3"
    + "_h_size_"
    + str(M3_HIDDEN_SIZE)
    + "_t_layers_"
    + str(TOKEN_NUM_LAYERS)
    + "_p_layers_"
    + str(PATCH_NUM_LAYERS)
    + "_p_size_"
    + str(PATCH_SIZE)
    + "_p_length_"
    + str(PATCH_LENGTH)
    + "_lr_"
    + str(M3_LEARNING_RATE)
    + "_batch_"
    + str(M3_BATCH_SIZE)
    + "_mask_"
    + str(M3_MASK_RATIO)
    + ".pth"
)
M3_LOGS_PATH = M3_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")

CLAMP3_TRAIN_JSONL = "<YOUR_TRAINING_JSONL_FILE>"
CLAMP3_EVAL_JSONL = "<YOUR_EVALUATION_JSONL_FILE>"

CLAMP3_HIDDEN_SIZE = 768
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_TEXT_LENGTH = 128

AUDIO_HIDDEN_SIZE = 768
AUDIO_NUM_LAYERS = 12
MAX_AUDIO_LENGTH = 128

CLAMP3_NUM_EPOCH = 100
CLAMP3_LEARNING_RATE = 1e-5
CLAMP3_BATCH_SIZE = 256
LOGIT_SCALE = 1

FREEZE_TEXT = False
TEXT_DROPOUT = True
CLAMP3_DETERMINISTIC = True
CLAMP3_LOAD_M3 = True
CLAMP3_WANDB_LOG = True
CLAMP3_LOAD_CKPT = True
SAVE_EVERY = 5

CLAMP3_WEIGHTS_PATH = (
    "weights_clamp3_saas"
    + "_h_size_"
    + str(CLAMP3_HIDDEN_SIZE)
    + "_t_model_"
    + TEXT_MODEL_NAME.replace("/", "_")
    + "_t_length_"
    + str(MAX_TEXT_LENGTH)
    + "_a_size_"
    + str(AUDIO_HIDDEN_SIZE)
    + "_a_layers_"
    + str(AUDIO_NUM_LAYERS)
    + "_a_length_"
    + str(MAX_AUDIO_LENGTH)
    + "_s_size_"
    + str(M3_HIDDEN_SIZE)
    + "_s_layers_"
    + str(PATCH_NUM_LAYERS)
    + "_p_size_"
    + str(PATCH_SIZE)
    + "_p_length_"
    + str(PATCH_LENGTH)
    + ".pth"
)
CLAMP3_LOGS_PATH = CLAMP3_WEIGHTS_PATH.replace("weights", "logs").replace(
    "pth",
    "txt",
)
