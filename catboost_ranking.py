import os
import json
import warnings
from pathlib import Path
from datetime import datetime
import pickle
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from catboost import CatBoostRanker, Pool

from utils import set_random_seeds, get_best_weights
from parameters import *
from setlist import SetList
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings, split_dataset
from embedding_head import EmbeddingHead, iterate_dataset_with_model
from baselines.bm25_inference import BM25Custom, tokenize


SIMILARITY_FUNCTION_DIM_0 = torch.nn.CosineSimilarity(dim=0)
SIMILARITY_FUNCTION_DIM_1 = torch.nn.CosineSimilarity(dim=1)
set_random_seeds(600)
# TODO: pickle dataset splits before training, so that at the table creation time we refer to the same dataset


def create_tabular_datasets():
    train_dict = json.load(open('Dataset/task1_train_labels_2024_reduced.json'))
    train_preprocessing_folder_path = 'Dataset/translated_train'
    train_embeddings_folder_path = 'Dataset/gpt_embed_train'
    train_dict, val_dict = split_dataset(train_dict, split_ratio=SPLIT_RATIO)
    print(f'Building Dataset with split ratio {SPLIT_RATIO}...')

    train_group_id, train_features, train_labels = \
        get_tabular_features(train_dict, train_preprocessing_folder_path, train_embeddings_folder_path)
    val_group_id, val_features, val_labels = \
        get_tabular_features(val_dict, train_preprocessing_folder_path, train_embeddings_folder_path)

    normalization_model = train_normalization_model(train_features, val_features)
    train_features = normalization_model(train_features)
    val_features = normalization_model(val_features)

    test_dict = json.load(open('Dataset/task1_test_labels_2024_reduced.json'))
    test_preprocessing_folder_path = 'Dataset/translated_test'
    test_embeddings_folder_path = 'Dataset/gpt_embed_test'

    test_group_id, test_features, test_labels = \
        get_tabular_features(test_dict, test_preprocessing_folder_path, test_embeddings_folder_path)
    test_features = normalization_model(test_features)

    dataset = {
        'train': (train_group_id, train_features, train_labels),
        'val': (val_group_id, val_features, val_labels),
        'test': (test_group_id, test_features, test_labels),
        'normalization_model': normalization_model
    }

    with open('Dataset/tabular_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)


class Normalizer:
    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        normalized_x = []
        n_cols = len(self.params)
        for i in range(len(x)):
            normalized_row = [(x[i][j] - self.params[j][0]) / math.sqrt(self.params[j][1] ** 2 + 1e-3) for j in range(n_cols)]
            normalized_x.append(normalized_row)
        return normalized_x


def train_normalization_model(train_features, val_features):
    # Concatenate train and val features
    features = train_features + val_features
    # Train normalization model
    n_cols = len(features[0])
    params = []
    for i in range(n_cols):
        mean = sum([x[i] for x in features]) / len(features)
        std = math.sqrt(sum([(x[i] - mean) ** 2 for x in features]) / (len(features) - 1))
        params.append((mean, std))

    return Normalizer(params)


def get_tabular_features(files_dict, preprocessing_folder_path, embeddings_folder_path):
    embeddings = get_gpt_embeddings(folder_path=embeddings_folder_path, selected_dict=files_dict)

    query_dataset = QueryDataset(embeddings, files_dict)
    document_dataset = DocumentDataset(embeddings, files_dict)
    q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    d_dataloader = DataLoader(document_dataset, batch_size=128, shuffle=False)

    recall_model = EmbeddingHead(hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
    recall_model.load_weights(get_best_weights('recall'))

    f1_model = EmbeddingHead(hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
    f1_model.load_weights(get_best_weights('val_f1_score'))
    f1_model.eval()

    files = []
    for key, values in files_dict.items():
        files.append(key)
        files.extend(values)
    files = SetList(files)

    d_preprocessed_corpus = [(filename, open(os.path.join(preprocessing_folder_path, filename), encoding='utf-8')
                              .read()) for filename in files]
    d_tokenized_corpus = tokenize(d_preprocessed_corpus)

    q_preprocessed_corpus = [(filename, open(os.path.join(preprocessing_folder_path, filename), encoding='utf-8')
                              .read()) for filename in list(files_dict.keys())]
    q_tokenized_corpus = tokenize(q_preprocessed_corpus)

    bag_model = BM25Custom(corpus=d_tokenized_corpus, k1=3.0, b=1.0)

    top_n_scores = [x for x in iterate_dataset_with_model(recall_model,
                                                          (q_dataloader, d_dataloader),
                                                          pe_cutoff=20,
                                                          score_function=SIMILARITY_FUNCTION_DIM_1,
                                                          score_iterator_mode=True)]

    pbar = tqdm(total=len(q_preprocessed_corpus), desc='Creating tabular dataset')

    group_id = []
    features = []
    labels = []
    for q_idx, recall_model_scores in enumerate(top_n_scores):
        q_name = list(files_dict.keys())[q_idx]
        q_emb = embeddings[q_name].to('cuda')
        bm25_scores = bag_model.get_scores(q_tokenized_corpus[q_idx])

        for d_name, _ in recall_model_scores:
            d_emb = embeddings[d_name].to('cuda')

            f1_model_q, f1_model_d = f1_model(q_emb, doc=d_emb)
            f1_model_score = float(SIMILARITY_FUNCTION_DIM_0(f1_model_q, f1_model_d))
            gpt_score = float(SIMILARITY_FUNCTION_DIM_0(q_emb, d_emb))
            bm25_score = float(bm25_scores[files.index(d_name)])

            group_id.append(q_idx)
            features.append([f1_model_score, gpt_score, bm25_score])

        pe_names = list(set(files_dict[q_name]))
        d_names = [x[0] for x in recall_model_scores]
        # TODO: it could happen that the labels computed for a group are all zeros. Check if this breaks catboost
        labels.extend([1 if x in pe_names else 0 for x in d_names])

        pbar.update(1)
    pbar.close()

    return group_id, features, labels


if __name__ == '__main__':
    create_tabular_datasets()

    # Load the pools
    with open('Dataset/tabular_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_group_id, train_features, train_labels = dataset['train']
    val_group_id, val_features, val_labels = dataset['val']
    test_group_id, test_features, test_labels = dataset['test']
    normalization_model = dataset['normalization_model']

    train_pool = Pool(data=train_features, label=train_labels, group_id=train_group_id)
    val_pool = Pool(data=val_features, label=val_labels, group_id=val_group_id)
    test_pool = Pool(data=test_features, label=test_labels, group_id=test_group_id)

    train_pool.set_feature_names(['f1_model', 'gpt', 'bm25'])
    val_pool.set_feature_names(['f1_model', 'gpt', 'bm25'])
    test_pool.set_feature_names(['f1_model', 'gpt', 'bm25'])


    print('Done!')
