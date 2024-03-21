import os
import json
import warnings
from pathlib import Path
import pickle
import math

import numpy as np
import torch
from tqdm import tqdm
from catboost import CatBoostRanker, Pool

from utils import set_random_seeds, get_best_weights
from parameters import *
from dataset import get_gpt_embeddings, create_dataloaders, split_dataset
from embedding_head import EmbeddingHead, iterate_dataset_with_model

SIMILARITY_FUNCTION_DIM_0 = torch.nn.CosineSimilarity(dim=0)
SIMILARITY_FUNCTION_DIM_1 = torch.nn.CosineSimilarity(dim=1)
set_random_seeds(600)


def create_tabular_datasets():
    _, train_dataloader = create_dataloaders('train', invert=True)
    _, val_dataloader = create_dataloaders('train', invert=False)
    train_dict, val_dict = split_dataset(load=True)

    train_bm25_scores = pickle.load(open('Dataset/bm25_train_results.pkl', 'rb'))
    test_bm25_scores = pickle.load(open('Dataset/bm25_test_results.pkl', 'rb'))
    train_tfidf_scores = pickle.load(open('Dataset/tfidf_train_results.pkl', 'rb'))
    test_tfidf_scores = pickle.load(open('Dataset/tfidf_test_results.pkl', 'rb'))

    train_group_id, train_features, train_labels, train_predicted_evidences = \
        get_tabular_features(qd_dataloader=train_dataloader,
                             files_dict=train_dict,
                             embeddings=get_gpt_embeddings(
                                 folder_path=Path.joinpath(Path('Dataset'), Path('gpt_embed_train')),
                                 selected_dict=train_dict),
                             bm25_scores=train_bm25_scores,
                             tfidf_scores=train_tfidf_scores
                             )
    val_group_id, val_features, val_labels, val_predicted_evidences = \
        get_tabular_features(qd_dataloader=val_dataloader,
                             files_dict=val_dict,
                             embeddings=get_gpt_embeddings(
                                 folder_path=Path.joinpath(Path('Dataset'), Path('gpt_embed_train')),
                                 selected_dict=val_dict),
                             bm25_scores=train_bm25_scores,
                             tfidf_scores=train_tfidf_scores
                             )

    normalization_model = train_normalization_model(train_features, val_features)
    train_features = normalization_model(train_features)
    val_features = normalization_model(val_features)

    _, test_dataloader = create_dataloaders('test')
    test_dict = json.load(open(Path.joinpath(Path('Dataset'), Path('task1_test_labels_2024.json'))))
    test_group_id, test_features, test_labels, test_predicted_evidences = \
        get_tabular_features(qd_dataloader=test_dataloader,
                             files_dict=test_dict,
                             embeddings=get_gpt_embeddings(
                                 folder_path=Path.joinpath(Path('Dataset'), Path('gpt_embed_test')),
                                 selected_dict=test_dict),
                             bm25_scores=test_bm25_scores,
                             tfidf_scores=test_tfidf_scores
                             )
    test_features = normalization_model(test_features)

    dataset = {
        'train': (train_group_id, train_features, train_labels, train_predicted_evidences),
        'val': (val_group_id, val_features, val_labels, val_predicted_evidences),
        'test': (test_group_id, test_features, test_labels, test_predicted_evidences),
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
            normalized_row = [(x[i][j] - self.params[j][0]) / math.sqrt(self.params[j][1] ** 2 + 1e-3) for j in
                              range(n_cols)]
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


def get_tabular_features(qd_dataloader, files_dict, embeddings, bm25_scores, tfidf_scores):
    q_dataloader, d_dataloader = qd_dataloader

    recall_model = EmbeddingHead(hidden_units=RECALL_HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
    recall_model.load_weights(get_best_weights('recall'))

    f1_model = EmbeddingHead(hidden_units=F1_HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
    f1_model.load_weights(get_best_weights('val_f1_score'))
    f1_model.eval()

    top_n_scores = [x for x in iterate_dataset_with_model(recall_model, qd_dataloader,
                                                          pe_cutoff=PE_CUTOFF,
                                                          score_function=SIMILARITY_FUNCTION_DIM_1,
                                                          score_iterator_mode=True,
                                                          verbose=True)]

    pbar = tqdm(total=len(q_dataloader.dataset), desc='Creating tabular dataset')

    predicted_evidences = {}
    group_id = []
    features = []
    labels = []
    for q_idx, recall_model_scores in enumerate(top_n_scores):
        q_name = list(files_dict.keys())[q_idx]
        q_emb = embeddings[q_name].to('cuda')
        # bm25_scores = bag_model.get_scores(q_tokenized_corpus[q_idx])

        for d_name, _ in recall_model_scores:
            d_emb = embeddings[d_name].to('cuda')

            f1_model_q, f1_model_d = f1_model(q_emb, doc=d_emb)
            f1_model_score = float(SIMILARITY_FUNCTION_DIM_0(f1_model_q, f1_model_d))
            f1_model_dot_score = float(torch.dot(f1_model_q, f1_model_d))
            gpt_score = float(SIMILARITY_FUNCTION_DIM_0(q_emb, d_emb))
            gpt_dot_score = float(torch.dot(q_emb, d_emb))
            # bm25_score = float(bm25_scores[files.index(d_name)])
            bm25_score = bm25_scores[(q_name, d_name)]
            tfidf_score = tfidf_scores[(q_name, d_name)]

            group_id.append(q_idx)
            features.append([f1_model_score, f1_model_dot_score, gpt_score, gpt_dot_score, bm25_score, tfidf_score])

        pe_names = list(set(files_dict[q_name]))
        d_names = [x[0] for x in recall_model_scores]
        # TODO: it could happen that the labels computed for a group are all zeros. Check if this breaks catboost
        labels.extend([1 if x in pe_names else 0 for x in d_names])
        predicted_evidences[q_name] = d_names

        pbar.update(1)
    pbar.close()

    return group_id, features, labels, predicted_evidences


def apply_cutoff(arr):
    mask = arr < arr[0] * CATBOOST_SIM_RATIO
    # mask = arr < CATBOOST_THRESHOLD
    arr[mask] = 0
    arr[~mask] = 1
    return arr


def convert_scores(scores, targets):
    results = np.stack((scores, targets), axis=2)
    results[:, ::-1, :].sort(axis=1)
    # Shape: (n_queries, pe_cutoff, 2)
    np.apply_along_axis(apply_cutoff, 1, results[:, :, 0])
    return results


def get_metrics(results, missed_positives):
    results = results.reshape(-1, 2)
    tp = np.sum(results[:, 0] * results[:, 1])
    fp = np.sum(results[:, 0] * (1 - results[:, 1]))
    fn = np.sum((1 - results[:, 0]) * results[:, 1]) + missed_positives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def get_missed_positives(e_dict, mode='test'):
    train_dict, val_dict = split_dataset(load=True)
    test_dict = json.load(open(Path.joinpath(Path('Dataset'), Path('task1_test_labels_2024.json'))))

    if mode == 'train':
        json_dict = train_dict
    elif mode == 'val':
        json_dict = val_dict
    elif mode == 'test':
        json_dict = test_dict
    else:
        raise ValueError('Invalid mode')

    missed_positives = 0
    for q in json_dict.keys():
        for e in set(json_dict[q]):
            if e not in e_dict[q]:
                missed_positives += 1
    return missed_positives


if __name__ == '__main__':
    # create_tabular_datasets()
    # quit()

    # Load the pools
    with open('Dataset/tabular_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_group_id, train_features, train_labels, train_predicted_evidences = dataset['train']
    val_group_id, val_features, val_labels, val_predicted_evidences = dataset['val']
    test_group_id, test_features, test_labels, test_predicted_evidences = dataset['test']
    normalization_model = dataset['normalization_model']

    train_pool = Pool(data=train_features, label=train_labels, group_id=train_group_id)
    val_pool = Pool(data=val_features, label=val_labels, group_id=val_group_id)
    test_pool = Pool(data=test_features, label=test_labels, group_id=test_group_id)
    whole_pool = Pool(data=train_features + val_features, label=train_labels + val_labels,
                      group_id=train_group_id + [x + 1024 for x in val_group_id])

    train_pool.set_feature_names(['f1_model', 'f1_model_dot', 'gpt', 'gpt_dot', 'bm25', 'tfidf'])
    val_pool.set_feature_names(['f1_model', 'f1_model_dot', 'gpt', 'gpt_dot', 'bm25', 'tfidf'])
    test_pool.set_feature_names(['f1_model', 'f1_model_dot', 'gpt', 'gpt_dot', 'bm25', 'tfidf'])
    whole_pool.set_feature_names(['f1_model', 'f1_model_dot', 'gpt', 'gpt_dot', 'bm25', 'tfidf'])

    # Train the model
    model = CatBoostRanker(loss_function='YetiRank', task_type='CPU')
    model.fit(whole_pool, verbose=True)
    model.save_model('catboost_model.bin')

    # Y = model._predict(test_pool, 'Probability', 0, 0, -1, None,
    #                    parent_method_name='predict')[:, 0]
    Y = model.predict(test_pool)

    n_queries = len(set(test_group_id))
    Y = Y.reshape(n_queries, PE_CUTOFF)
    gt = np.array(test_labels).reshape(n_queries, PE_CUTOFF)

    res = convert_scores(Y, gt)
    metrics = get_metrics(res, missed_positives=get_missed_positives(test_predicted_evidences, mode='test'))
    print(f'Precision: {metrics[0]:.6f}, Recall: {metrics[1]:.6f}, F1 score: {metrics[2]:.6f}')

    print(model.get_feature_importance(test_pool, type='PredictionValuesChange', prettified=True))

    print('Done!')
