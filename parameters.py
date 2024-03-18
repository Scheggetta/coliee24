# Preprocessing
PREPROCESSING_DATASET_TYPE = 'train'  # 'train' or 'test'
FRENCH_THRESHOLD = 0.4

# Embedding head
# TODO: maybe add dropout or change architecture
EMB_IN = 1536
EMB_OUT = 50            # 50
HIDDEN_UNITS = 650      # recall: 240, f1_score: 650

# Torch Dataset
# TODO: maybe the negative samples should be taken from the most similar documents according to BM25 (and also increase sample size)
SAMPLE_SIZE = 50              # 50 - Negative samples per query
SPLIT_RATIO = 0.8             # 0.8
HARD_NEGATIVE_MINING = False  # False

# Loss function
PE_WEIGHT = None           # None
COSINE_LOSS_MARGIN = 0.4   # 0.37

# LR scheduler
# TODO: AdamW instead of Adam?
LR = 0.001          # 0.001
FACTOR = 0.1        # 0.1
THRESHOLD = 0.001   # 0.001
PATIENCE = 5        # 5
COOLDOWN = 3        # 3

# Cutoff hyperparameters
DYNAMIC_CUTOFF = True         # False
PE_CUTOFF = 20                 # 5
MAX_DOCS = 10                 # 10
RATIO_MAX_SIMILARITY = 0.95   # 0.95

# BM25
BM25_TOP_N = 5   # 5

# Regularization
DROPOUT_RATE = 0.0  # 0.2

# Catboost
CATBOOST_SIM_RATIO = 0.9  # 0.9


def set_sample_size(n):
    global SAMPLE_SIZE
    SAMPLE_SIZE = n


def get_sample_size():
    return SAMPLE_SIZE

