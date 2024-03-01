import nltk
import os
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.special import softmax, kl_div, rel_entr
import json
from pathlib import Path
import random
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
from argostranslate import package, translate
import re
import shutil

pd.set_option('display.max_columns', None)
package.install_from_path('fr_en.argosmodel')


def split_queries_evidences(dataset_folder='Dataset/task1_train_files_2024', query_folder='Dataset/Train_Queries', evidence_folder='Dataset/Train_Evidence', labels_file='Dataset/task1_train_labels_2024.json'):
    file = open(labels_file)
    dict = json.load(file)
    os.mkdir(query_folder)
    for f in dict.keys():
        if Path.joinpath(Path(dataset_folder), Path(f)).exists():
            shutil.copy(Path.joinpath(Path(dataset_folder), Path(f)),
                        Path.joinpath(Path(query_folder), Path(f)))
    os.mkdir(evidence_folder)
    for l in dict.values():
        for f in l:
            if Path.joinpath(Path(dataset_folder), Path(f)).exists():
                shutil.copy(Path.joinpath(Path(evidence_folder), Path(f)),
                            Path.joinpath(Path(evidence_folder), Path(f)))


def get_tokenizer():
    return RegexpTokenizer(r'[a-zA-Z]\w+')


def get_stop_words():
    eng_swrds = nltk.corpus.stopwords.words('english')
    fr_swrds = nltk.corpus.stopwords.words('french')
    return eng_swrds + fr_swrds


def get_directory_vocabulary(directory='Dataset/task1_train_files_2024'):
    stop_words = get_stop_words()
    tokenizer = get_tokenizer() # RegexpTokenizer(r'\w+')
    for file in os.listdir(directory):
        cumulative_text = []
        with open(f'{directory}/{file}', 'r', encoding='utf-8') as f:
            text = f.read().lower()
            text = tokenizer.tokenize(text)
            cumulative_text += [word for word in text if word not in stop_words]
    return set(cumulative_text)


def get_file_vocabulary(file):
    stop_words = get_stop_words()
    tokenizer = get_tokenizer()  # RegexpTokenizer(r'\w+')
    cumulative_text = []
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        text = tokenizer.tokenize(text)
        cumulative_text += [word for word in text if word not in stop_words]
    return set(cumulative_text)


def compare_vocabs(vocab1, vocab2):
    return vocab1.intersection(vocab2)


def compare_documents(vocab1, vocab2):
    return len(vocab1.intersection(vocab2))/len(vocab1.union(vocab2))  # if len(vocab1.union(vocab2)) > 0 else 0


def get_directory_words_frequency(directory='Dataset/task1_train_files_2024', max_words=50):
    stop_words = get_stop_words()
    tokenizer = get_tokenizer()  # RegexpTokenizer(r'\w+')
    cumulative_text = []
    for file in os.listdir(directory):
        with open(f'{directory}/{file}', 'r', encoding='utf-8') as f:
            text = f.read().lower()
            text = tokenizer.tokenize(text)
            cumulative_text += [word for word in text if word not in stop_words]
    return dict(nltk.FreqDist(cumulative_text).most_common(max_words))


def get_file_words_frequency(file, max_words=50):
    stop_words = get_stop_words()
    tokenizer = get_tokenizer()  # RegexpTokenizer(r'\w+')
    cumulative_text = []
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        text = tokenizer.tokenize(text)
        cumulative_text += [word for word in text if word not in stop_words]
    return dict(nltk.FreqDist(cumulative_text).most_common(max_words))


def compare_file_freqs(vocabulary, file1, file2, max_words=50):
    freq1 = get_file_words_frequency(file1, max_words)
    translation_dict = {k: v for k, v in zip(vocabulary, range(len(vocabulary)))}
    np_freq1 = np.zeros(len(vocabulary))
    for k, v in freq1.items():
        if k in translation_dict.keys():
            np_freq1[translation_dict[k]] = v
    np_freq1 = softmax(np_freq1)
    freq2 = get_file_words_frequency(file2, max_words)
    np_freq2 = np.zeros(len(vocabulary))
    for k, v in freq2.items():
        if k in translation_dict.keys():
            np_freq2[translation_dict[k]] = v
    np_freq2 = softmax(np_freq2)
    return sum(rel_entr(np_freq1, np_freq2))  # kl_div(np_freq1, np_freq2)


def compute_cosine_similarity(vocabulary, freq1, freq2):
    vector1 = np.array([freq1[word] if word in freq1.keys() else 0 for word in vocabulary])  # / np.linalg.norm(np.array([freq1[word] if word in freq1.keys() else 0 for word in vocabulary]))
    vector2 = np.array([freq2[word] if word in freq2.keys() else 0 for word in vocabulary])  # / np.linalg.norm(np.array([freq2[word] if word in freq2.keys() else 0 for word in vocabulary]))
    return cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))


def get_evidences(query):
    labels = json.load(open('Dataset/task1_train_labels_2024.json', 'r'))
    return labels[query]


def pick_random_evidence(notch_evidences, n=5):
    evidence_set = set(os.listdir('Dataset/Train_Evidence'))
    evidence_set = evidence_set - set(notch_evidences)
    random_choices = []
    for i in range(n):
        choice = random.choice(list(evidence_set))
        random_choices.append(choice)
        evidence_set = evidence_set - set(choice)
    return random_choices


def get_language(file):
    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        return detector.detect_multiple_languages_of(text), len(text)


def get_french_percentage(query_dir_switch=True):
    dir_val = 'Dataset/Train_Queries' if query_dir_switch else 'Dataset/Train_Evidence'
    perc_list = []
    text_len_sum = 0
    for f in os.listdir(dir_val):
        language, text_len = get_language(Path.joinpath(Path(dir_val), Path(f)))
        file_french_list = []
        for lan in language:
            if lan.language == Language.FRENCH:
                file_french_list.append(lan.end_index - lan.start_index)
                text_len_sum += text_len
        perc_list.append((sum(file_french_list)/text_len, f))
    dataset_french_perc = sum([perc for perc, _ in perc_list])/len(perc_list)
    return perc_list, dataset_french_perc


def compare_french_english_script(path_to_file):
    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    installed_languages = translate.get_installed_languages()
    translation_fr_en = installed_languages[1].get_translation(installed_languages[0])
    file_french_translations = []
    with open(path_to_file, 'r', encoding='utf-8') as f:
        text = f.read()
        french_to_translate = detector.detect_multiple_languages_of(text)
        for lan in french_to_translate:
            if lan.language == Language.FRENCH:
                french_text = text[lan.start_index:lan.end_index]
                translated_text = translation_fr_en.translate(french_text)
                if translated_text != french_text:
                    file_french_translations.append((french_text, translated_text))
    return file_french_translations
    # pitfalls: the language detection is a hard tarsk because we are with English and French which are very similar.
    # Moreover, both the language detector and translation tool are robust enough to take account of several kind of typos
    # then sometimes the translation is affected by the fact that the detection tool takes classifies English sentence as French


def get_parenthesis_freqs_dataset(directory='Dataset/task1_train_files_2024'):
    freq_dict = dict()
    for file in os.listdir(directory):
        with open(Path.joinpath(Path(directory), Path(file)), 'r', encoding='utf-8') as f:
            text = f.read()
            list_found_keys = re.findall(r'\(.*?\)', text)
            for key in list_found_keys:
                if key in freq_dict.keys():
                    freq_dict[key] += 1
                else:
                    freq_dict[key] = 1
    return freq_dict
    # No significant information can be extracted from the parenthesis


def get_bracket_freqs_dataset(directory='Dataset/task1_train_files_2024'):
    freq_dict = dict()
    for file in os.listdir(directory):
        with open(Path.joinpath(Path(directory), Path(file)), 'r', encoding='utf-8') as f:
            text = f.read()
            list_found_keys = re.findall(r'\[.*?\]', text)
            for key in list_found_keys:
                if key in freq_dict.keys():
                    freq_dict[key] += 1
                else:
                    freq_dict[key] = 1
    return freq_dict
    # numbers: 1222
    # strings: 6404
    # There are more than 1000 numbers describing paragraphs
    # So it's difficult to discriminate between dates and paragraph numbers
    # Among the strings there are:
    #       notes (in English and in French)
    #       names of the documents
    #       names of people related to that case
    #       acronyms
    #       references to other documents
    #       references to other cases
    #       references to other paragraphs
    #       references to other laws
    #       references to footnotes
    #       references to citations
    #       references to omitted fragments
    # dict(sorted([(f,freqs[f]) for f in freqs.keys() if not(f[1:-1].isnumeric())], key=lambda x: x[1], reverse=True))
    # dict(sorted([(f,freqs[f]) for f in freqs.keys() if f[1:-1].isnumeric()], key=lambda x: x[1], reverse=True)


def analyse_dataset(dataset_folder, query_folder, evidence_folder, freq_threshold=500):
    if not (Path(query_folder).exists() and Path(evidence_folder).exists()):
        split_queries_evidences(dataset_folder)
    general_vocabulary = get_directory_vocabulary()
    avg_iou_ev_list, avg_cos_sim_ev_list, avg_kl_div_ev_list = [], [], []
    avg_iou_re_list, avg_cos_sim_re_list, avg_kl_div_re_list = [], [], []
    for query in os.listdir(query_folder):
        evidences = get_evidences(query)
        random_evidence = pick_random_evidence(evidences)
        vocab_query = get_file_vocabulary(f'{query_folder}/{query}')
        iou_ev_list , cos_sim_ev_list, kl_div_ev_list = [], [], []
        for e in evidences:
            vocab_evidence = get_file_vocabulary(f'{evidence_folder}/{e}')
            iou_ev = compare_documents(vocab_query, vocab_evidence)
            cos_sim_ev = compute_cosine_similarity(general_vocabulary, get_file_words_frequency(f'{query_folder}/{query}', freq_threshold), get_file_words_frequency(f'{evidence_folder}/{e}', freq_threshold))
            kl_div_ev = compare_file_freqs(general_vocabulary, f'{query_folder}/{query}', f'{evidence_folder}/{e}')

            iou_ev_list.append(iou_ev)
            cos_sim_ev_list.append(cos_sim_ev)
            kl_div_ev_list.append(kl_div_ev)
        avg_iou_ev_list.append(np.array(iou_ev_list).mean())
        avg_cos_sim_ev_list.append(np.array(cos_sim_ev_list).mean())
        avg_kl_div_ev_list.append(np.array(kl_div_ev_list).mean())
        iou_re_list, cos_sim_re_list, kl_div_re_list = [], [], []
        for re in random_evidence:
            vocab_evidence = get_file_vocabulary(f'{evidence_folder}/{re}')
            iou_re = compare_documents(vocab_query, vocab_evidence)
            cos_sim_re = compute_cosine_similarity(general_vocabulary, get_file_words_frequency(f'{query_folder}/{query}', freq_threshold), get_file_words_frequency(f'{evidence_folder}/{re}', freq_threshold))
            kl_div_re = compare_file_freqs(general_vocabulary, f'{query_folder}/{query}', f'{evidence_folder}/{re}')

            iou_re_list.append(iou_re)
            cos_sim_re_list.append(cos_sim_re)
            kl_div_re_list.append(kl_div_re)
        avg_iou_re_list.append(np.array(iou_re_list).mean())
        avg_cos_sim_re_list.append(np.array(cos_sim_re_list).mean())
        avg_kl_div_re_list.append(np.array(kl_div_re_list).mean())

    return pd.DataFrame({'query': os.listdir(query_folder),
                                       'average_iou_evidences': avg_iou_ev_list,
                                       'average_iou_random_evidences': avg_iou_re_list,
                                       'average_cos_sim_evidences': avg_cos_sim_ev_list,
                                       'average_cos_sim_random_evidences': avg_cos_sim_re_list,
                                       'average_kl_div_evidences': avg_kl_div_ev_list,
                                       'average_kl_div_random_evidences': avg_kl_div_re_list})



# TODO: Translate the queries and the evidence files to a common language (English)
# TODO: Determine the English/French ratio in the dataset
# TODO: Determine if the French sentences are the translation of the English ones


