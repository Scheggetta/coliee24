import json
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score

from parameters import *
from dataset import create_dataloaders
from embedding_head import iterate_dataset_with_model
from utils import set_random_seeds

if __name__ == '__main__':
    _, qd_dataloader = create_dataloaders(PREPROCESSING_DATASET_TYPE)


    def score_function(query, doc):
        return torch.ones((len(doc)))


    iters = iterate_dataset_with_model(model=None,
                                       validation_dataloader=qd_dataloader,
                                       score_function=score_function,
                                       pe_cutoff=PE_CUTOFF,
                                       verbose=True
                                       )
    res = next(iters)
    print(f'precision: {res[4]}, recall: {res[5]}, f1_score: {res[6]}')
