import os
import random
import shutil
import json
from functools import reduce
from colorama import Style
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


def average_negative_evidences(tensor_list):
    total_length = 0
    num_tensors = len(tensor_list)

    for tensor in tensor_list:
        total_length += len(tensor)

    if num_tensors == 0:
        return 0  # return 0 if the list is empty to avoid division by zero

    return total_length / num_tensors


def get_best_weights(metric='val_f1_score', mode=max):
    weights = [x for x in os.listdir('Checkpoints') if x.endswith('.pt') and metric in x]
    paths = sorted(weights, key=lambda x: float(x.split(sep='_')[-1][:-3]))
    assert len(paths) > 0, "No weights found! Please train the model on the chosen metric first!"
    best_path = paths[-1] if mode == 'max' else paths[0]
    return Path.joinpath(Path('Checkpoints'), Path(best_path))


def fancy_print(scores, name, color):
    print(f"{color}{name} Test set results: \n"
          f"    Precision: {Style.BRIGHT}{scores[0]:.4f}{Style.NORMAL}, "
          f"Recall: {Style.BRIGHT}{scores[1]:.4f}{Style.NORMAL}, "
          f"F1 Score: {Style.BRIGHT}{scores[2]:.4f}{Style.NORMAL}!")


def convert_dict(original_dict, n):
    new_dict = {}
    for key, value in original_dict.items():
        if key[0] in new_dict:
            new_dict[key[0]].append((key[1], value))
        else:
            new_dict[key[0]] = [(key[1], value)]
    for key, value in new_dict.items():
        new_dict[key] = [x[0] for x in sorted(value, key=lambda x: x[1])[::-1][:n]]
    return new_dict


def compute_metrics(query_dict):
    gt_dict = json.load(open('Dataset/task1_test_labels_2024.json', 'r'))
    TP = 0
    FP = 0
    FN = 0
    for key in gt_dict:
        for pos_doc in gt_dict[key]:
            if pos_doc in query_dict[key]:
                TP += 1
            else:
                FN += 1
        for pos_doc in query_dict[key]:
            if pos_doc not in gt_dict[key]:
                FP += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score

