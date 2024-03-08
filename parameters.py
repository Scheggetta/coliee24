# Preprocessing
PREPROCESSING_DATASET_TYPE = 'test'  # 'train' or 'test'

# Embedding head
EMB_IN = 1536
EMB_OUT = 50            # 50
HIDDEN_UNITS = 256      # 256

# Torch Dataset
SAMPLE_SIZE = 15        # 15 - Negative samples per query

# Loss function
PE_WEIGHT = None          # None
COSINE_LOSS_MARGIN = 0.4  # 0.5

# LR scheduler
LR = 0.001          # 0.001
FACTOR = 0.1        # 0.1
THRESHOLD = 0.001    # 0.001
PATIENCE = 5        # 3
COOLDOWN = 3        # 3

# Cutoff hyperparameters
DYNAMIC_CUTOFF = True        # True
PE_CUTOFF = 5                # 5
MAX_DOCS = 10                # 10
RATIO_MAX_SIMILARITY = 0.95   # 0.9
