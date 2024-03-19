from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from pathlib import Path


def get_stop_words():
    eng_swrds = nltk.corpus.stopwords.words('english')
    fr_swrds = nltk.corpus.stopwords.words('french')
    return eng_swrds + fr_swrds


def read_texts(path):
    texts = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def tfidf_similarity(query, evidences):
    filtered_query = ' '.join(list(filter(lambda x: x not in get_stop_words(), query.lower())))
    text_list = [open(file, 'r', encoding='utf-8').read() for file in os.listdir(evidences)[:150]]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_list)
    query_vector = vectorizer.transform([query])
    pass


if __name__ == '__main__':
    query = open('Dataset/Train_Queries_regex/000002.txt', 'r', encoding='utf-8').read()
    evidences = 'Dataset/Train_Evidence_regex'
    tfidf_similarity(query, evidences)


# TODO: implemet the tfidf function (define a way to select the terms to describe a document)