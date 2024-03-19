import torch

from parameters import *
from dataset import create_dataloaders
from utils import set_random_seeds
from embedding_head import iterate_dataset_with_model


if __name__ == '__main__':
    set_random_seeds(1984)
    _, qd_dataloader = create_dataloaders(PREPROCESSING_DATASET_TYPE)


    def score_function(query, doc):
        return torch.rand((len(doc),)) * 2 - 1


    iters = iterate_dataset_with_model(model=None,
                                       validation_dataloader=qd_dataloader,
                                       score_function=score_function,
                                       pe_cutoff=PE_CUTOFF,
                                       verbose=True
                                       )
    res = next(iters)
    print(f'precision: {res[4]}, recall: {res[5]}, f1_score: {res[6]}')
