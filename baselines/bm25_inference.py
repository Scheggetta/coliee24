import os
import json
import random

import nltk
from nltk.tokenize import RegexpTokenizer
import re
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
import torch
from torcheval.metrics.functional import binary_f1_score

from parameters import *
from dataset import split_dataset
from setlist import SetList


seed = 62
print(f'Setting seed to {seed}')
random.seed(seed)
np.random.seed(seed)


def lowercase(text):
    return text.lower()


def remove_stop_words(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])


def stemming(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])


def tokenize(texts):
    corpus = [re.sub(r'\[\d{1,4}\]', '', text) for _, text in texts]
    corpus = [lowercase(text) for text in corpus]
    corpus = [remove_stop_words(text) for text in corpus]
    stemmer = nltk.stem.PorterStemmer()
    corpus = [stemming(text, stemmer) for text in corpus]

    tokenizer = RegexpTokenizer(r'[a-zA-Z]\w+')
    tokenized_corpus = [tokenizer.tokenize(text) for text in corpus]
    return tokenized_corpus


class BM25Custom(BM25Okapi):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_top_n_indexes(self, query, n):
        scores = self.get_scores(query)
        scores = np.argsort(scores)[::-1]
        top_n = scores[:n]

        # TODO: check that the first element is not the query itself

        # TODO
        # threshold = similarities[0][1] * RATIO_MAX_SIMILARITY
        # predicted_pe = [x for x in similarities if x[1] >= threshold]
        # predicted_pe = predicted_pe[:MAX_DOCS] if len(predicted_pe) > MAX_DOCS else predicted_pe

        return top_n


if __name__ == '__main__':
    folder = 'Dataset/translated_%s' % PREPROCESSING_DATASET_TYPE
    json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
    train_dict, val_dict = split_dataset(json_dict, split_ratio=0.01)

    val_files = []
    for key, values in val_dict.items():
        val_files.append(key)
        val_files.extend(values)
    val_files = SetList(val_files)

    files = val_files
    files_dict = val_dict

    d_preprocessed_corpus = [(filename, open(os.path.join(folder, filename)).read()) for filename in files]
    d_tokenized_corpus = tokenize(d_preprocessed_corpus)

    q_preprocessed_corpus = [(filename, open(os.path.join(folder, filename)).read()) for filename in list(files_dict.keys())]
    q_tokenized_corpus = tokenize(q_preprocessed_corpus)

    model = BM25Custom(corpus=d_tokenized_corpus, k1=3.0, b=1.0)

    pbar = tqdm(total=len(q_preprocessed_corpus), desc='BM25 Inference')

    f1 = 0.0
    i = 0

    correctly_retrieved_cases = 0
    retrieved_cases = 0
    relevant_cases = 0

    for q_name, _ in q_preprocessed_corpus:
        q_text = q_tokenized_corpus[i]
        pe = val_dict[q_name]
        pe_idxs = [files.index(x) for x in pe]

        relevant_cases += len(pe)

        indexes = model.get_top_n_indexes(q_text, n=BM25_TOP_N)
        predicted_pe_names = [d_preprocessed_corpus[idx][0] for idx in indexes if idx != files.index(q_name)]
        predicted_pe_idxs = [files.index(x) for x in predicted_pe_names]

        gt = torch.zeros(len(files))
        gt[pe_idxs] = 1
        targets = torch.zeros(len(files))
        targets[predicted_pe_idxs] = 1

        correctly_retrieved_cases += len(gt[(gt == 1) & (targets == 1)])
        retrieved_cases += len(targets[targets == 1])
        precision = correctly_retrieved_cases / retrieved_cases
        recall = correctly_retrieved_cases / relevant_cases
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        f1 += binary_f1_score(gt, targets)

        i += 1
        pbar.set_description(f'f1: {f1_score:.4f}')
        pbar.update()
