# Preprocessing
PREPROCESSING_DATASET_TYPE = 'test'  # 'train' or 'test'
SEUIL = 0.4  # French Threshold

# Embedding head
EMB_IN = 1536
EMB_OUT = 50            # 50
HIDDEN_UNITS = 240      # recall: 240, f1_score: 650
RECALL_HIDDEN_UNITS = 240
F1_HIDDEN_UNITS = 650

# Torch Dataset
SAMPLE_SIZE = 50              # 50 - Negative samples per query
SPLIT_RATIO = 0.8             # 0.8
HARD_NEGATIVE_MINING = False  # False

# Loss function
PE_WEIGHT = None           # None
COSINE_LOSS_MARGIN = 0.4   # 0.37

# LR scheduler
LR = 0.001          # 0.001
FACTOR = 0.1        # 0.1
THRESHOLD = 0.001   # 0.001
PATIENCE = 5        # 5
COOLDOWN = 3        # 3

# Cutoff hyperparameters
DYNAMIC_CUTOFF = False         # False
PE_CUTOFF = 25                 # 5
MAX_DOCS = 10                 # 10
RATIO_MAX_SIMILARITY = 0.95   # 0.95

# BM25
BM25_TOP_N = 5   # 5

# TFIDF
TFIDF_TOP_N = 10  # 5

# Regularization
DROPOUT_RATE = 0.0  # 0.2

# Catboost
CATBOOST_DYNAMIC_CUTOFF = True    # True
CATBOOST_SIM_RATIO = 0.65           # 0.65
CATBOOST_STATIC_CUTOFF = 5         # 5

CATBOOST_WHOLE_DATASET = False      # False - If True, the whole training dataset is used for training;
                                    # otherwise the training dataset is split into train and validation sets
CATBOOST_LOAD_MODEL = True                 # False
CATBOOST_DATE_FILTER = True                # False
CATBOOST_DATE_RANGE_FILTER = (1970, 2020)  # (1970, 2020)



def set_sample_size(n):
    global SAMPLE_SIZE
    SAMPLE_SIZE = n


def get_sample_size():
    return SAMPLE_SIZE

