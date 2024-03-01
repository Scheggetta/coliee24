import torch as pt
import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import copy

from memory_profile import total_size
from model import Model

import utils

utils.check_split()  # Checking if the local dataset is split correctly

Roby = AutoModel.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
q_embed = tokenizer([open(Path.joinpath(Path('Dataset/Train_Queries'), x),
                          encoding='utf-8').read() for x in os.listdir('Dataset/Train_Queries')],
                    padding=True, truncation=True, max_length=512, return_tensors='pt')

K = 1  # Number of employed queries, selected in appearing order
batch_size = 256
set_size = len(os.listdir('Dataset/Train_Evidence'))  # Total number of possible evidences
q_embed = utils.take_first_k(q_embed, K)
model = Model('roberta-base').to('cuda')
scores = []  # Dictionaries with 'F1', 'precision', 'recall', 'n_guesses'
for query_n in range(K):
    print(f"Executing query n.{query_n + 1}")
    query = copy.deepcopy(q_embed).to('cuda')  # transferring tensors to GPU
    query['input_ids'] = query['input_ids'][query_n:query_n + 1].to('cuda')
    query['attention_mask'] = query['attention_mask'][query_n:query_n + 1].to('cuda')
    str_q = os.listdir('Dataset/Train_Queries')[query_n]
    true_evidences = json.load(open('Dataset/task1_train_labels_2024.json'))[str_q]
    res = np.array([])

    evidence = np.array(os.listdir('Dataset/Train_Evidence'))
    np.random.shuffle(evidence)
    for e_batch in range(set_size // batch_size + 1):
        print(f'Loading evidences from n.{e_batch * batch_size}')
        lim = (e_batch + 1) * batch_size if (e_batch + 1) * batch_size < set_size else -1
        evidence_set = evidence[e_batch * batch_size: lim]
        e_tokens = tokenizer([open(Path.joinpath(Path('Dataset/Train_Evidence'), x),
                                   encoding='utf-8').read() for x in evidence_set],
                             padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        similarities = model(query, e_tokens)
        # print(np.array(similarities.cpu())[similarities.cpu() > 0.995])
        res = np.append(res, np.array(evidence_set)[similarities.cpu() > 0.99])
    print(f'Got {len(res)} results')
    # print(res)
    scores.append(utils.baseline_F1(res, np.array(true_evidences)))

print(scores)

# TODO: consider of using a mixed approach (bag of words and embeddings) [select the most meaningful parts of the text and then use embeddings to compare them]
#       Find a way to get the most meaningful parts of the text
#       or use a mixed approach (embeddings and regex) [use regex to remove the most common parts of the text and then use embeddings to compare them]


# how to find the most meaningful parts of the text:
#   - use regex to remove the most common parts of the text (consider that too) [reason sentence-wise]
#   - consider the sentences in which the intersection of the vocabularies is the highest (consider training a neural network to do that)
#   - maybe consider a mixture of both:
#       - consider the most common parts of the text in terms of sentences and take the percentage of similarity subset of the text
#       - remove the most common parts of the text
#       - compare the remaining part with the embeddings
