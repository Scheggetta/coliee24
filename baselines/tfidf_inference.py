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


def import_corpus(train=True):
    train_flag = 'train' if train else 'test'
    return [open(f'Dataset/translated_{train_flag}/{x}', 'r', encoding='utf-8').read()
            for x in os.listdir(f'Dataset/translated_{train_flag}')]


def train_model(corpus):
    vectorizer = TfidfVectorizer(stop_words='English',
                                 lowercase=True,
                                 max_features=1500,
                                 ngram_range=(1, 2),
                                 tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X.toarray()


def predict(queries, documents_names, similarity_function, tf_idf_table, thr=None):
    queries = list(filter(lambda x: x in documents_names, queries))
    queries_idxs = list(map(lambda x: documents_names.index(x), queries))
    results = dict()
    cnt = 1
    for i, q in zip(queries_idxs, queries):
        print(f'query idx: {cnt} out of {len(queries)}')
        cnt += 1
        for i_doc, d_name in enumerate(documents_names):
            if i != i_doc:
                similarity_metric = similarity_function(tf_idf_table[i, :], tf_idf_table[i_doc, :])
                if thr:
                    results[(q, d_name)] = 1 if similarity_metric > thr else 0
                else:
                    results[(q, d_name)] = similarity_metric
    return results


def get_prediction_data(json_path, documents_folder):
    map_dict = json.load(open(json_path))
    queries = list(map_dict.keys())
    documents_names = os.listdir(documents_folder)
    return map_dict, queries, documents_names


def tf_idf(save_model=True, save_results=True, split_ratio=0.8, threshold=None, load_model=True,
           save_path=Path('Dataset')):
    # Threshold is in charge of crisp predictions instead of metrics results
    corpus = import_corpus()

    train_corpus = corpus
    # split_ratio = 0.8
    if split_ratio:
        train_corpus, validation_corpus = corpus[:int(len(corpus) * split_ratio)], corpus[
                                                                                   int(len(corpus) * split_ratio):]

    if (load_model and os.path.exists(Path.joinpath(save_path, Path('tf_idf_model.pkl')))
            and os.path.exists(Path.joinpath(save_path, Path('tf_idf_X.pkl')))):
        model = pickle.load(open(Path.joinpath(save_path, Path('tf_idf_model.pkl')), 'rb'))
        X = pickle.load(open(Path.joinpath(save_path, Path('tf_idf_X.pkl')), 'rb'))
    else:
        model, X = train_model(train_corpus)

    if save_model:
        pickle.dump(model, open(Path.joinpath(save_path, Path('tf_idf_model.pkl')), 'wb'))
        pickle.dump(X, open(Path.joinpath(save_path, Path('tf_idf_X.pkl')), 'wb'))

    similarity_function = lambda x, y: x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # threshold = 0.5

    train_dict, train_queries, documents_names = \
        get_prediction_data('Dataset/task1_train_labels_2024.json',
                            'Dataset/translated_train')

    extent = int(len(documents_names) * split_ratio) if split_ratio else len(documents_names)
    documents_names = documents_names[:extent]
    train_results = predict(train_queries, documents_names, similarity_function, X, threshold)

    if split_ratio:
        val_dict, val_queries, documents_names = \
            get_prediction_data('Dataset/task1_train_labels_2024.json',
                                'Dataset/translated_train')
        X_val = model.transform(validation_corpus).toarray()
        extent = int(len(documents_names) * split_ratio) if split_ratio else len(documents_names)
        documents_names = documents_names[extent:]
        val_results = predict(val_queries, documents_names, similarity_function, X_val, threshold)
    else:
        val_results = None

    test_dict, test_queries, test_documents_names = \
        get_prediction_data('Dataset/task1_test_labels_2024.json',
                            'Dataset/translated_test')

    test_corpus = import_corpus(train=False)
    X_test = model.transform(test_corpus).toarray()
    test_results = predict(test_queries, test_documents_names, similarity_function, X_test, threshold)

    if save_results:
        pickle.dump([train_results, val_results, test_results], open(save_path, 'wb'))

    return train_results, val_results, test_results


def compute_f1_score(results, json_dict):
    predicted = [results[i, j] for i, j in results.keys()]
    ground_truth = [1 if j in json_dict[i] else 0 for i, j in results.keys()]
    return f1_score(ground_truth, predicted)


if __name__ == '__main__':
    train_results, _, test_results = tf_idf(threshold=0.5, save_results=False,
                                            split_ratio=0.8, load_model=True,
                                            )#save_path=Path.joinpath(Path('Dataset'), Path('tfidf_baseline')))
    # f1 = compute_f1_score(test_results, json.load(open('Dataset/task1_test_labels_2024.json')))
    # print(f"F1 score: {f1}!!!")

