import os
import random
import shutil
import json
from functools import reduce
from pathlib import Path
import numpy as np
import torch


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


def set_random_seeds(seed):
    print(f'Setting seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# q = (1, 2, 3)
# e = (2, 3, 3)
# q = torch.Tensor([[[1, 2, 1], [1, 2, 1]]])
# e = torch.Tensor([[[-1, 2, 3], [-2, -1, 1], [0, 0, 1]], [[-1, -1, -1], [0, -1, 1], [0, 0, 0]]])
# x = torch.matmul(q, torch.transpose(e, 1, 2))
#
# x2 = torch.einsum('ijk,lmk->jlm', [q, e])
#
# q = torch.rand(1, 512, 768)
# e = torch.rand(10, 512, 768)
# x = torch.matmul(q, torch.transpose(e, 1, 2))
# x2 = torch.einsum('ijk,ljk->l', [q, e])
#
# q = torch.rand(1, 512, 768)
# e = torch.rand(10, 512, 768)
# q = torch.mean(q, dim=1)
# e = torch.mean(e, dim=1)
# cos = torch.nn.CosineSimilarity()
# x = cos(q, e)
#
# # q = (1, 5)
# # e = (3, 5)
# q = torch.Tensor([[1, 2, 3, 4, 5]])
# attn = torch.Tensor([1, 1, 0, 0, 0])
# q = q * attn
# q = torch.squeeze(q)
# e = torch.Tensor([[1, 2, -1, 4, -2], [1, -1, 3, 4, -1], [-1, -1, -1, -1, 5]])
# x = torch.matmul(q, torch.transpose(e, 0, 1))
# x = cos(q, e)

