PROJECT         = 'cough-detection'
# DATA CONFIG ============================
DATA_PATH       = './data'
DATA_SAVE_PATH  = './data/preprocessed'
DATASET         = 'warm-up-8k'
DVERSION        = ':latest'
SAVE_PATH       = './model'

VALID_RATE      = 0.2

# MODEL CONFIG ===========================
RUN_NAME        = 'demo'
INPUT_SIZE      = 640
BATCH_SIZE      = 32
NUM_WORKER      = 0

EPOCH           = 5
LR              = 1e-3

# PREDICTION CONFIG ======================
DATA_PRED_PATH  = './data/test'
PRED_SAVE       = './data/test/prediction'