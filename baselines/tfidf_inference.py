from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import os
import json
from functools import reduce
import numpy as np
import pickle
from sklearn.metrics import f1_score
from pathlib import Path
from bm25_inference import tokenize_parallel, tokenize_corpus_from_dict
from parameters import *
import torch
from tqdm import tqdm


class TfidfCustom:
    def __init__(self, train_corpus, test_corpus):
        self.model = TfidfVectorizer(analyzer=lambda x: x)
        self.X_train = self.model.fit_transform(train_corpus).toarray()
        self.X_test = self.model.transform(test_corpus).toarray()

    def get_scores(self, query):
        return self.model.transform([query]).toarray().dot(self.X_train.T).flatten()

    def get_test_scores(self, query):
        return self.model.transform([query]).toarray().dot(self.X_test.T).flatten()

    def get_top_n_indexes(self, query, n, return_scores=False):
        scores = self.get_scores(query)
        indexes = np.argsort(scores)[::-1]
        top_n = indexes[:n]

        if return_scores:
            return top_n, scores[top_n]

        return top_n

    def get_test_top_n_indexes(self, query, n, return_scores=False):
        scores = self.get_test_scores(query)
        scores = np.argsort(scores)[::-1]
        top_n = scores[:n]

        if return_scores:
            return top_n, scores[top_n]

        return top_n


def iterate_dataset_with_tfidf(model,
                               files,
                               files_dict,
                               q_pr_corpus,
                               q_tk_corpus,
                               mode: str,
                               return_results=False,
                               get_top_n=True
                               ):
    assert mode in ['train', 'test'], 'Invalid mode'

    results = dict()
    pbar = tqdm(total=len(q_pr_corpus), desc='TFIDF %s' % mode)

    f1 = 0.0
    i = 0

    correctly_retrieved_cases = 0
    retrieved_cases = 0
    relevant_cases = 0

    for q_name, _ in q_pr_corpus:
        q_text = q_tk_corpus[i]

        if return_results:
            if mode == 'train':
                scores = model.get_scores(q_text)
            elif mode == 'test':
                scores = model.get_test_scores(q_text)
            for d_name, score in zip(files, scores):
                if d_name != q_name:
                    results[(q_name, d_name)] = score

        if get_top_n:
            pe = files_dict[q_name]
            pe_idxs = [files.index(x) for x in pe]

            relevant_cases += len(pe)

            if mode == 'train':
                indexes = model.get_top_n_indexes(q_text, n=TFIDF_TOP_N+1)
            elif mode == 'test':
                indexes = model.get_test_top_n_indexes(q_text, n=TFIDF_TOP_N+1)
            predicted_pe_idxs = [idx for idx in indexes if idx != files.index(q_name)]

            gt = torch.zeros(len(files))
            gt[pe_idxs] = 1
            targets = torch.zeros(len(files))
            targets[predicted_pe_idxs] = 1

            correctly_retrieved_cases += len(gt[(gt == 1) & (targets == 1)])
            retrieved_cases += len(targets[targets == 1])
            precision = correctly_retrieved_cases / retrieved_cases
            recall = correctly_retrieved_cases / relevant_cases
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            pbar.set_description(f'pre: {precision:.4f} - '
                                 f'rec: {recall:.4f} - '
                                 f'f1: {f1_score:.4f}')

        i += 1
        pbar.update()
    pbar.close()

    if return_results:
        return results


if __name__ == '__main__':
    train_folder = Path.joinpath(Path('Dataset'), Path('translated_train'))
    train_dict = json.load(open(Path.joinpath(Path('Dataset'), Path('task1_train_labels_2024.json'))))
    test_folder = Path.joinpath(Path('Dataset'), Path('translated_test'))
    test_dict = json.load(open(Path.joinpath(Path('Dataset'), Path('task1_test_labels_2024.json'))))

    train_files, train_d_pr_corpus, train_d_tk_corpus, train_q_pr_corpus, train_q_tk_corpus = \
        tokenize_corpus_from_dict(train_dict, train_folder)
    test_files, test_d_pr_corpus, test_d_tk_corpus, test_q_pr_corpus, test_q_tk_corpus = \
        tokenize_corpus_from_dict(test_dict, test_folder)

    model = TfidfCustom(train_d_tk_corpus, test_d_tk_corpus)

    train_results = iterate_dataset_with_tfidf(model, train_files, train_dict, train_q_pr_corpus, train_q_tk_corpus, 'train', return_results=True, get_top_n=False)
    test_results = iterate_dataset_with_tfidf(model, test_files, test_dict, test_q_pr_corpus, test_q_tk_corpus, 'test', return_results=True, get_top_n=False)
    pickle.dump(train_results, open('Dataset/tfidf_train_results.pkl', 'wb'))
    pickle.dump(test_results, open('Dataset/tfidf_test_results.pkl', 'wb'))