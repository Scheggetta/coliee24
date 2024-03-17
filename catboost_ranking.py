import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from catboost import CatBoostRanker

from utils import set_random_seeds, average_negative_evidences, get_best_weights
from parameters import *
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings, split_dataset
from embedding_head import EmbeddingHead, iterate_dataset_with_model


set_random_seeds(600)


def create_tabular_dataset():
    json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
    train_dict, val_dict = split_dataset(json_dict, split_ratio=SPLIT_RATIO)
    print(f'Building Dataset with split ratio {SPLIT_RATIO}...')

    training_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_%s' % PREPROCESSING_DATASET_TYPE,
                                             selected_dict=train_dict)
    validation_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_%s' % PREPROCESSING_DATASET_TYPE,
                                               selected_dict=val_dict)

    train_query_dataset = QueryDataset(training_embeddings, train_dict)
    train_document_dataset = DocumentDataset(training_embeddings, train_dict)
    train_q_dataloader = DataLoader(train_query_dataset, batch_size=1, shuffle=False)
    train_d_dataloader = DataLoader(train_document_dataset, batch_size=128, shuffle=False)

    val_query_dataset = QueryDataset(validation_embeddings, val_dict)
    val_document_dataset = DocumentDataset(validation_embeddings, val_dict)
    val_q_dataloader = DataLoader(val_query_dataset, batch_size=1, shuffle=False)
    val_d_dataloader = DataLoader(val_document_dataset, batch_size=128, shuffle=False)

    model = EmbeddingHead(hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
    model.load_weights(get_best_weights('recall'))

    print('Creating tabular dataset...')

    model_scores = [x for x in iterate_dataset_with_model(model,
                                                          (train_q_dataloader, train_d_dataloader),
                                                          score_function=...,
                                                          score_iterator_mode=True)]
