PROJECT         = 'cough-detection'
# DATA CONFIG ============================
DATA_PATH       = './data'
DATA_SAVE_PATH  = './data/preprocessed'
DATASET         = 'unsilence-warm-up-8k'
DVERSION        = 'latest'
SAVE_PATH       = './model'

VALID_RATE      = 0.2

# PROCESSING MEL =========================
SR              = 8000
FRAME_LENGTH    = 8064
N_FFT           = 1024
N_MFCC          = 128
HOP_LENGTH_FFT  = 512
DURATION        = 8150
N_MELS          = 128

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