import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score

from parameters import *
from dataset import create_dataloaders
from utils import set_random_seeds
from embedding_head import iterate_dataset_with_model


if __name__ == '__main__':
    set_random_seeds(1984)
    train_dataloader, qd_dataloader = create_dataloaders(PREPROCESSING_DATASET_TYPE)
    cs_fn = torch.nn.CosineSimilarity(dim=1)

    iters = iterate_dataset_with_model(model=None,
                                       validation_dataloader=qd_dataloader,
                                       score_function=cs_fn,
                                       pe_cutoff=PE_CUTOFF,
                                       verbose=True
                                       )
    res = next(iters)
    print(f'precision: {res[4]}, recall: {res[5]}, f1_score: {res[6]}')

