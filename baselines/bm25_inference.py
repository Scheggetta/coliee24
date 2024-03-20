import os
import json
import random
from multiprocessing import Pool, cpu_count

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
from pathlib import Path
from utils import set_random_seeds


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


def tokenize_one(text, _stemmer, _tokenizer):
    text = re.sub(r'\[\d{1,4}\]', '', text)
    text = lowercase(text)
    text = remove_stop_words(text)
    text = stemming(text, _stemmer)
    return _tokenizer.tokenize(text)


def tokenize_parallel(texts):
    stemmer = nltk.stem.PorterStemmer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]\w+')

    # Since no text depends on another, we can parallelize the tokenization
    pool = Pool(cpu_count())
    tokenized_corpus = pool.starmap(tokenize_one, [(text, stemmer, tokenizer) for _, text in texts])
    return tokenized_corpus


class BM25Custom(BM25Okapi):
    def __init__(self, test_corpus=None, **kwargs):
        super().__init__(**kwargs)
        if test_corpus:
            self.test_corpus_size = 0
            self.test_doc_len = None
            self.test_doc_freqs = None
            self._initialize_test_corpus(test_corpus)

    def _initialize_test_corpus(self, test_corpus):
        test_doc_len = []
        test_doc_freqs = []

        for document in test_corpus:
            test_doc_len.append(len(document))

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            test_doc_freqs.append(frequencies)

            self.test_corpus_size += 1

        self.test_doc_len = np.array(test_doc_len)
        self.test_doc_freqs = test_doc_freqs

    def get_test_scores(self, query):
        score = np.zeros(self.test_corpus_size)
        doc_len = self.test_doc_len
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.test_doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_test_top_n_indexes(self, query, n):
        scores = self.get_test_scores(query)
        scores = np.argsort(scores)[::-1]
        top_n = scores[:n]

        return top_n

    def get_top_n_indexes(self, query, n):
        scores = self.get_scores(query)
        scores = np.argsort(scores)[::-1]
        top_n = scores[:n]

        return top_n

    def get_top_n_indexes_scores(self, query, n):
        scores = self.get_scores(query)
        top_n = [(idx, score) for idx, score in enumerate(scores)]
        top_n = sorted(top_n, key=lambda x: x[1], reverse=True)[:n]

        return top_n


if __name__ == '__main__':
    if PREPROCESSING_DATASET_TYPE == 'train':
        _, files_dict = split_dataset(load=True)
    else:
        files_dict = json.load(open(Path.joinpath(Path('Dataset'), Path(f'task1_test_labels_2024.json'))))
    folder = Path.joinpath(Path('Dataset'), Path(f'translated_{PREPROCESSING_DATASET_TYPE}'))

    # files_dict = {key: value for n, (key, value) in enumerate(files_dict.items()) if n < 10}

    def tokenize_corpus():
        pass

    files = []
    for key, values in files_dict.items():
        files.append(key)
        files.extend(values)
    files = SetList(files)

    d_preprocessed_corpus = [(filename, open(os.path.join(folder, filename), encoding='utf-8').read())
                             for filename in files]
    d_tokenized_corpus = tokenize_parallel(d_preprocessed_corpus)

    q_preprocessed_corpus = [(filename, open(os.path.join(folder, filename), encoding='utf-8').read())
                             for filename in list(files_dict.keys())]
    q_tokenized_corpus = tokenize_parallel(q_preprocessed_corpus)

    model = BM25Custom(corpus=d_tokenized_corpus, k1=3.0, b=1.0, test_corpus=None)

    pbar = tqdm(total=len(q_preprocessed_corpus), desc='BM25 Inference')

    f1 = 0.0
    i = 0

    correctly_retrieved_cases = 0
    retrieved_cases = 0
    relevant_cases = 0

    for q_name, _ in q_preprocessed_corpus:
        q_text = q_tokenized_corpus[i]
        pe = files_dict[q_name]
        pe_idxs = [files.index(x) for x in pe]

        relevant_cases += len(pe)

        indexes = model.get_top_n_indexes(q_text, n=BM25_TOP_N+1)
        # indexes = model.get_test_top_n_indexes(q_text, n=BM25_TOP_N+1)
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

        i += 1
        pbar.set_description(f'pre: {precision:.4f} - '
                             f'rec: {recall:.4f} - '
                             f'f1: {f1_score:.4f}')
        pbar.update()
    pbar.close()
