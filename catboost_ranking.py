import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from catboost import CatBoostRanker

from utils import set_random_seeds, get_best_weights
from parameters import *
from setlist import SetList
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings, split_dataset
from embedding_head import EmbeddingHead, iterate_dataset_with_model
from baselines.bm25_inference import BM25Custom, tokenize


SIMILARITY_FUNCTION = torch.nn.CosineSimilarity(dim=1)
set_random_seeds(600)


def create_tabular_dataset():
    json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
    folder = Path.joinpath(Path('Dataset'), Path(f'translated_preprocessed_{PREPROCESSING_DATASET_TYPE}'))
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
    model.load_weights(get_best_weights('val_f1_score'))

    print('Creating tabular dataset...')

    model_scores = [x for x in iterate_dataset_with_model(model,
                                                          (train_q_dataloader, train_d_dataloader),
                                                          pe_cutoff=20,
                                                          score_function=SIMILARITY_FUNCTION,
                                                          score_iterator_mode=True)]

    GPT_scores = [x for x in iterate_dataset_with_model(None,
                                                        (train_q_dataloader, train_d_dataloader),
                                                        pe_cutoff=20,
                                                        score_function=SIMILARITY_FUNCTION,
                                                        score_iterator_mode=True)]

    train_files = []
    for key, values in train_dict.items():
        train_files.append(key)
        train_files.extend(values)
    train_files = SetList(train_files)

    files = train_files
    files_dict = train_dict

    d_preprocessed_corpus = [(filename, open(os.path.join(folder, filename), encoding='utf-8').read()) for filename in files]
    d_tokenized_corpus = tokenize(d_preprocessed_corpus)

    q_preprocessed_corpus = [(filename, open(os.path.join(folder, filename), encoding='utf-8').read()) for filename in
                             list(files_dict.keys())]
    q_tokenized_corpus = tokenize(q_preprocessed_corpus)

    bag_model = BM25Custom(corpus=d_tokenized_corpus, k1=3.0, b=1.0)
    okapi_indexes = [bag_model.get_top_n_indexes_scores(q_text, n=BM25_TOP_N + 1)[1:] for q_text in q_tokenized_corpus]
    okapi_scores = []
    for q in okapi_indexes:
        okapi_scores.append([(d_preprocessed_corpus[idx][0], score) for idx, score in q])

    print('Creating tabular dataset, part 2. This may take a while... (2/2)')


if __name__ == '__main__':
    create_tabular_dataset()
    print('Done!')
