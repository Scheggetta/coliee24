import torch as pt
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import shutil

from memory_profile import total_size
from model import Model


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


Roby = AutoModel.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
e_embed = tokenizer([open(Path.joinpath(Path('Dataset/Train_Evidence'), x)).read() for x in os.listdir('Dataset/Train_Evidence')], padding=True, truncation=True, max_length=512, return_tensors='pt')


# print(len(os.listdir('Dataset/Train_Evidence')))
# print(total_size(q_embed.data))


def take_first_k(embed, k):
    embed['input_ids'] = embed['input_ids'][:k]
    embed['attention_mask'] = embed['attention_mask'][:k]
    return embed

e_embed = take_first_k(e_embed, 10)
model = Model('roberta-base')
result = model(e_embed)


pass
