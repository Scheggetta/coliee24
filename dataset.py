import os
import pickle
import random
from math import ceil
from pathlib import Path

import torch
from torch.utils.data import Dataset

from setlist import SetList
from parameters import *
from utils import set_random_seeds


class TrainingDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings = embeddings
        self.json_dict = json_dict

        queries_names = [key for key in json_dict if key in embeddings]
        self.all_evidences = sorted(list(set([evidence for query_name in queries_names for evidence in json_dict[query_name]])))

        self.queries = []
        for q_name in queries_names:
            self.queries.append((q_name, embeddings[q_name]))

        self.evidences = []
        for q_name in queries_names:
            single_evidences = torch.empty((0, EMB_IN))
            evidences = json_dict[q_name]
            for e_name in evidences:
                single_evidences = torch.cat((single_evidences, embeddings[e_name].unsqueeze(0)), dim=0)
            self.evidences.append((evidences, single_evidences))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query_name = self.queries[index][0]
        query = self.queries[index][1]
        evidence_names = self.evidences[index][0]
        evidence = self.evidences[index][1]

        excluded_documents = evidence_names.copy()
        excluded_documents.append(query_name)

        sample_space = self.all_evidences.copy()
        for el in excluded_documents:
            try:
                sample_space.remove(el)
            except ValueError:
                pass

        negative_evidences_names = random.sample(list(sample_space), get_sample_size())

        negative_evidences = torch.empty((0, EMB_IN))
        for e_name in negative_evidences_names:
            negative_evidences = torch.cat((negative_evidences, self.embeddings[e_name].unsqueeze(0)), dim=0)

        return query, evidence, negative_evidences


class QueryDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings = embeddings
        self.json_dict = json_dict

        queries_names = [key for key in json_dict if key in embeddings]
        self.queries = [(q_name, embeddings[q_name]) for q_name in queries_names]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]


class DocumentDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings_dict = embeddings
        self.json_dict = json_dict

        self.embeddings = []
        self.masked_query = None
        self.masked_evidences = None

        for key in embeddings:
            self.embeddings.append((key, embeddings[key]))

    def __len__(self):
        return len(self.embeddings)

    def mask(self, query_name):
        if self.masked_query:
            raise ValueError('A document is already masked')
        q_embedding = self.embeddings_dict[query_name]
        self.masked_evidences = self.json_dict[query_name]

        self.masked_query = (query_name, q_embedding)
        self.embeddings.remove((query_name, q_embedding))

    def restore(self):
        if self.masked_query:
            self.embeddings.append(self.masked_query)
            self.masked_query = None
            self.masked_evidences = None
        else:
            raise ValueError('No element to restore')

    def __getitem__(self, index):
        return self.embeddings[index]

    def get_indexes(self, filenames):
        return [self.embeddings.index((x, self.embeddings_dict[x])) for x in filenames]


def custom_collate_fn(batch: list):
    queries = torch.stack([x[0] for x in batch])
    evidences = [x[1] for x in batch]
    negative_evidences = torch.stack([x[2] for x in batch])

    return queries, evidences, negative_evidences


def get_gpt_embeddings(folder_path: Path, selected_dict: dict):
    embeddings = {}

    files = []
    for key, values in selected_dict.items():
        files.append(key)
        files.extend(values)
    files = SetList(files)

    # Check if any selected evidence does not have a related query
    for file in files:
        if file in selected_dict:
            # `file` is a query
            evidences = selected_dict[file]
            assert any(evidence in files for evidence in evidences), f'No evidence for query {file}'
        else:
            # `file` is an evidence
            found_query = False
            for key in selected_dict:
                if file in selected_dict[key]:
                    found_query = True
                    break
            assert found_query, f'No query for evidence {file}'

    for file in files:
        with open(os.path.join(folder_path, file), 'rb') as f:
            e = pickle.load(f)
            n_paragraphs = len(e) // EMB_IN
            assert len(e) % EMB_IN == 0

            e = torch.Tensor(e).view(n_paragraphs, EMB_IN).mean(dim=0)
            # e = torch.Tensor(e[:EMB_IN])
            embeddings[file] = e

    return embeddings


def split_dataset(json_dict=None, split_ratio=0.8, seed=42, save=True, load=False):
    if not load and json_dict is None:
        raise ValueError('json_dict is None and load is False')

    if load:
        with open('Dataset/train_dict.pkl', 'rb') as f:
            train_dict = pickle.load(f)
        with open('Dataset/val_dict.pkl', 'rb') as f:
            val_dict = pickle.load(f)
        return train_dict, val_dict

    set_random_seeds(seed)

    keys = list(json_dict.keys())
    random.shuffle(keys)

    train_size = ceil(len(json_dict) * split_ratio)
    train_dict = {key: json_dict[key] for key in keys[:train_size]}
    val_dict = {key: json_dict[key] for key in keys[train_size:]}

    if save:
        with open('Dataset/train_dict.pkl', 'wb') as f:
            pickle.dump(train_dict, f)
        with open('Dataset/val_dict.pkl', 'wb') as f:
            pickle.dump(val_dict, f)

    return train_dict, val_dict
