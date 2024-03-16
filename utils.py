import os
import random
import shutil
import json
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import re


def check_split():
    if not (Path('Dataset/Train_Queries').exists() and Path('Dataset/Train_Evidence').exists()):
        print("Separating Dataset...")
        file = open("Dataset/task1_train_labels_2024.json")
        dict = json.load(file)
        os.mkdir('Dataset/Train_Queries')
        for f in dict.keys():
            if Path.joinpath(Path("Dataset/task1_train_files_2024"), Path(f)).exists():
                shutil.copy(Path.joinpath(Path("Dataset/task1_train_files_2024"), Path(f)),
                            Path.joinpath(Path('Dataset/Train_Queries'), Path(f)))
        os.mkdir('Dataset/Train_Evidence')
        for l in dict.values():
            for f in l:
                if Path.joinpath(Path("Dataset/task1_train_files_2024"), Path(f)).exists():
                    shutil.copy(Path.joinpath(Path("Dataset/task1_train_files_2024"), Path(f)),
                                Path.joinpath(Path('Dataset/Train_Evidence'), Path(f)))


def check_queries_evidences_split():
    fl = open("Dataset/task1_train_labels_2024.json")
    js = json.load(fl)
    evidence_list = reduce(lambda x, y: x+y, js.values())
    return all([Path.joinpath(Path('Dataset/Train_Evidence'), Path(f)).exists() for f in evidence_list])


def take_first_k(embed, k):
    # TODO: delete `encodings` list from `embed` to free memory
    embed['input_ids'] = embed['input_ids'][:k]
    embed['attention_mask'] = embed['attention_mask'][:k]
    return embed


def baseline_F1(pred, gt):
    TP = len([x for x in pred if x in gt])
    FP = len([x for x in pred if x not in gt])
    FN = len([x for x in gt if x not in pred])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return {'F1': 2 * precision * recall / (precision + recall), 'precision': precision,
            'recall': recall, 'n_guesses': len(pred)}


def contrastive_loss_function(query, pe, ne, cosine_loss_margin=None, pe_weight=None):
    loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    loss = torch.tensor(0.0).to('cuda')
    for idx, query_el in enumerate(query):
        query_el = query_el.unsqueeze(0)
        pe_el = pe[idx]
        ne_el = ne[idx]

        query_loss = torch.tensor(0.0).to('cuda')
        ne_dot = torch.einsum('ij,ij->i', query_el, ne_el)
        targets = torch.zeros(len(ne_dot) + 1).to('cuda')
        targets[0] = 1

        for el in pe_el:
            pe_dot = torch.einsum('ij,j->', query_el, el)
            logits = torch.cat((pe_dot.unsqueeze(0), ne_dot))
            loss += loss_function(logits, targets)
        query_loss += query_loss / len(pe_el)

    return loss / len(query)


def dot_similarity(query, doc):
    return torch.einsum('ij,ij->i', query, doc)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



