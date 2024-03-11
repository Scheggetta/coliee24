# Preprocessing
PREPROCESSING_DATASET_TYPE = 'train'  # 'train' or 'test'
FRENCH_THRESHOLD = 0.4

# Embedding head
# TODO: maybe add dropout or change architecture
EMB_IN = 1536
EMB_OUT = 50            # 50
HIDDEN_UNITS = 256      # 256

# Torch Dataset
# TODO: maybe the negative samples should be taken from the most similar documents according to BM25 (and also increase sample size)
SAMPLE_SIZE = 400        # 15 - Negative samples per query

# Loss function
PE_WEIGHT = 0.5          # None
COSINE_LOSS_MARGIN = 0.0  # 0.5

# LR scheduler
# TODO: AdamW instead of Adam?
LR = 0.001          # 0.001
FACTOR = 0.1        # 0.1
THRESHOLD = 0.001    # 0.001
PATIENCE = 3        # 3
COOLDOWN = 3        # 3

# Cutoff hyperparameters
DYNAMIC_CUTOFF = False        # True
PE_CUTOFF = 5                # 5
MAX_DOCS = 10                # 10
RATIO_MAX_SIMILARITY = 0.9   # 0.9

# BM25
BM25_TOP_N = 5   # 5
