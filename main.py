import torch as pt
import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import shutil

from memory_profile import total_size
from model import Model
from functools import reduce


def check_queries_evidences_split():
    fl = open("Dataset/task1_train_labels_2024.json")
    js = json.load(fl)
    evidence_list = reduce(lambda x,y: x+y, js.values())
    return all([Path.joinpath(Path('Dataset/Train_Evidence'), Path(f)).exists() for f in evidence_list])


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



Roby = AutoModel.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
q_embed = tokenizer([open(Path.joinpath(Path('Dataset/Train_Queries'), x)).read() for x in os.listdir('Dataset/Train_Queries')], padding=True, truncation=True, max_length=512, return_tensors='pt')
e_embed = tokenizer([open(Path.joinpath(Path('Dataset/Train_Evidence'), x)).read() for x in os.listdir('Dataset/Train_Evidence')], padding=True, truncation=True, max_length=512, return_tensors='pt')


# print(len(os.listdir('Dataset/Train_Evidence')))
# print(total_size(q_embed.data))


def take_first_k(embed, k):
    # TODO: delete `encodings` list from `embed` to free memory
    embed['input_ids'] = embed['input_ids'][:k]
    embed['attention_mask'] = embed['attention_mask'][:k]
    return embed

q_embed = take_first_k(q_embed, 1)
e_embed = take_first_k(e_embed, 10)
model = Model('roberta-base')
result = model(q_embed, e_embed)

