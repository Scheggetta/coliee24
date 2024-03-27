import os
import json
from pathlib import Path
import random
import shutil
import pickle

import numpy as np
import pandas as pd
from scipy.special import softmax, kl_div, rel_entr
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from lingua import Language, LanguageDetectorBuilder
from argostranslate import package, translate
import umap
import umap.plot
import matplotlib.pyplot as plt

from parameters import *
from dataset import create_dataloaders
from embedding_head import EmbeddingHead, iterate_dataset_with_model
from utils import get_best_weights
from catboost_ranking import Normalizer


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


def compare_french_english_script(path_to_file, output_path):
    languages = [Language.ENGLISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    installed_languages = translate.get_installed_languages()
    translation_fr_en = installed_languages[1].get_translation(installed_languages[0])
    text = ''
    with open(path_to_file, 'r', encoding='utf-8') as f:
        text = f.read()
        french_to_translate = detector.detect_multiple_languages_of(text)
        for lan in french_to_translate:
            if lan.language == Language.FRENCH:
                french_text = text[lan.start_index:lan.end_index]
                translated_text = translation_fr_en.translate(french_text)
                if translated_text != french_text:
                    text = text[:lan.start_index] + translated_text + text[lan.end_index:]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)



    # pitfalls: the language detection is a hard task because we are with English and French which are very similar.
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


def find_paragraph1_occurrences(input_directory):
    found_files = []
    not_found_files = []
    for idx, file_name in enumerate(os.listdir(input_directory)):
        file_text = open(Path.joinpath(Path(input_directory), Path(file_name))).read()
        if len(re.findall(r'\[1\]', file_text, flags=re.IGNORECASE)) > 0:
            found_files.append(file_name)
        else:
            not_found_files.append(file_name)
    return found_files, not_found_files, len(found_files) + len(not_found_files)


def calculate_summary_statistics(folder):
    n_summaries = 0
    n_useless_summaries = 0
    for file in os.listdir(folder):
        with open(Path.joinpath(Path(folder), Path(file)), 'r') as f:
            text = f.read()
            summary_len = len(re.findall(r'Summary:?\n', text, flags=re.IGNORECASE))
            if summary_len > 0:
                n_summaries += summary_len
                if len(re.findall(r'This case is unedited', text, flags=re.IGNORECASE)) > 0:
                    n_useless_summaries += 1
    return n_summaries, n_useless_summaries, n_summaries / len(os.listdir(folder)), n_useless_summaries / len(
        os.listdir(folder))


def calculate_aliens_topic_statistics(folder):
    n_topics = 0
    n_topics_after_paragraph = 0
    for file in os.listdir(folder):
        if file == '025275.txt':
            continue
        with open(Path.joinpath(Path(folder), Path(file)), 'r') as f:
            text = f.read()
            text = re.split(r'\[1\]', text)

            before_paragraph = len(re.findall(r'.* - topic (\d+(\.{1,}\d+){0,})', text[0], flags=re.IGNORECASE))
            after_paragraph = len(re.findall(r'.* - topic (\d+(\.{1,}\d+){0,})', text[1], flags=re.IGNORECASE))
            n_topics += before_paragraph + after_paragraph
            n_topics_after_paragraph += after_paragraph

    return n_topics, n_topics_after_paragraph

def get_embedding_dict():
    embeddings_folder = 'Dataset/gpt_embed_train'
    embedding_dict = dict()
    paragraph_count = dict()
    for file in os.listdir(embeddings_folder):
        if file == 'backup':
            continue
        with open(Path.joinpath(Path(embeddings_folder), Path(file)), 'rb') as f:
            e = pickle.load(f)
            n_paragraphs = len(e) // EMB_IN
            assert len(e) % EMB_IN == 0
            e = torch.Tensor(e).view(n_paragraphs, EMB_IN)
            emb = e.mean(dim=0)
            paragraph_count[file] = e
            embedding_dict[file] = emb
    return embedding_dict, paragraph_count


def get_mapper(embedding_dict, save=False, save_path='Dataset/umap_mapper_7.pkl'):
    if Path('Dataset/umap_mapper_7.pkl').exists():
        return pickle.load(open('Dataset/umap_mapper_7.pkl', 'rb'))

    if save:
        mapper = umap.UMAP(metric='cosine', n_neighbors=15, min_dist=.1, n_epochs=1000).fit(list(embedding_dict.values()))
        with open(save_path, 'wb') as f:
            pickle.dump(mapper, f)
        return mapper
    else:
        return umap.UMAP(metric='cosine').fit(list(embedding_dict.values()))


def plot_umap(mapper, embedding_dict, paragraphs, query, trans_dict):
    query_map = mapper.transform(embedding_dict[query].reshape(1, -1))
    paragraphs_map = mapper.transform(paragraphs[query])
    evidences = trans_dict[query]
    # fig = plt.figure()
    # ax = fig.gca()
    # umap.plot.points(mapper)
    # umap.plot.connectivity(mapper, edge_bundling='hammer')
    umap.plot.diagnostic(mapper, diagnostic_type='pca')
    # umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
    # umap.plot.diagnostic(mapper, diagnostic_type='all')
    # umap.plot.diagnostic(mapper, diagnostic_type='vq')
    # ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], s=2)
    plt.scatter(query_map[:, 0], query_map[:, 1], s=2, color='r', label=f'{query}')
    colors = ['blue', 'orange', 'violet', 'brown']
    for par, col in zip(paragraphs_map, colors):
        plt.scatter(par[0], par[1], s=2, color=col)
        plt.annotate(f'{col}', xy=(par[0], par[1]), xytext=(par[0]-.05, par[1]-.05),
                    fontsize=4)
    for ev in evidences:
        evidence_map = mapper.transform(embedding_dict[ev].reshape(1, -1))
        # plt.scatter(evidence_map[:, 0], evidence_map[:, 1], s=2, color='m', label=f'{ev}')
        # plt.annotate(f'{ev}', xy=(evidence_map[:, 0], evidence_map[:, 1]), xytext=(evidence_map[:, 0]-.05, evidence_map[:, 1]-.05),
        #             fontsize=4)
    # plt.legend()
    plt.show()


def extract_dates():
    dates_dict = {}
    months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
                   'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
    month = '(?:' + '|'.join(months_list) + ')'
    for folder in [Path.joinpath(Path('Dataset'), Path(f)) for f in ['task1_train_files_2024', 'task1_test_files_2024']]:
        for filename, text in [(x, open(Path.joinpath(folder, Path(x)), encoding='utf-8').read()) for x in os.listdir(folder)]:
            dates = re.findall(month + r' \d{1,2}, \d{4}', text)
            dates += re.findall(r'\d{1,2} ' + month + r' \d{4}', text)
            timestamps = [date.split(' ')[-1] for date in dates]
            timestamps.sort(reverse=True)
            dates_dict[filename] = int(timestamps[0]) if timestamps else None
    json_path = Path.joinpath(Path('Dataset'), Path('dates.json'))
    with open(json_path, 'w') as f:
        json.dump(dates_dict, f, indent=4)
    return dates_dict


if __name__ == '__main__':
    # extract_dates()
    # quit(0)

    _, test_dataloader = create_dataloaders('test')
    q_dataloader, d_dataloader = test_dataloader

    model = EmbeddingHead(hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE)
    model.load_weights(get_best_weights('recall', mode='max'))
    model.eval()

    json_dict = json.load(open('Dataset/task1_test_labels_2024.json', 'r'))
    documents = []
    for key in json_dict.keys():
        documents.append(key)
        documents += json_dict[key]
    documents = list(set(documents))

    def get_reduced_embedding(_model, embedding):
        embedding = _model.relu(_model.linear1(embedding))
        return _model.linear2(embedding)

    reduced_embeddings = {d_name: get_reduced_embedding(model, d_dataloader.dataset.embeddings_dict[d_name])
                                                                .detach() for d_name in documents}
    _all_embs = list(reduced_embeddings.values())
    if not os.path.exists('Dataset/umap_mapper.pkl'):
        mapper = umap.UMAP(metric='cosine').fit(_all_embs)
        with open('Dataset/umap_mapper.pkl', 'wb') as f:
            pickle.dump(mapper, f)
    else:
        mapper = pickle.load(open('Dataset/umap_mapper.pkl', 'rb'))

    # plt.rcParams["figure.dpi"] = 1000
    # umap.plot.points(mapper, width=1000, height=1000)
    # plt.show()

    dataset = pickle.load(open('Dataset/tabular_dataset_split.pkl', 'rb'))
    test_group_id, test_features, test_labels, test_predicted_evidences = dataset['test']
    _all_queries = list(json_dict.keys())

    def separate_correctly_predicted_evidences(query, pr_e):
        sf = torch.nn.CosineSimilarity(dim=0)

        correct = []
        wrong = []
        for e in pr_e:
            if e in json_dict[query]:
                correct.append(e)
            else:
                wrong.append(e)

        false_negatives = list(set(json_dict[query]) - set(pr_e))
        return (correct, [reduced_embeddings[e] for e in correct], [sf(reduced_embeddings[query], reduced_embeddings[e]) for e in correct],
                wrong, [reduced_embeddings[e] for e in wrong], [sf(reduced_embeddings[query], reduced_embeddings[e]) for e in wrong],
                false_negatives, [reduced_embeddings[e] for e in false_negatives], [sf(reduced_embeddings[query], reduced_embeddings[e]) for e in false_negatives])

    selected_query = _all_queries[1]
    print('Selected query:', selected_query)
    (correct_l, correct_embs, correct_scores,
     wrong_l, wrong_embs, wrong_scores,
     false_negatives_l, false_negatives_embs, false_negatives_scores) = \
        separate_correctly_predicted_evidences(selected_query, test_predicted_evidences[selected_query])

    fig = plt.figure(figsize=(8, 8), dpi=800)
    ax = fig.add_subplot(111)
    umap.plot.points(mapper, ax=ax)

    if len(wrong_embs) > 0:
        wrong_embs_proj = mapper.transform(wrong_embs)
        for idx, w in enumerate(wrong_embs_proj):
            plt.scatter(w[0], w[1], s=2, color='red')
            plt.annotate(wrong_l[idx] + f' - {wrong_scores[idx]:.4f}', xy=(w[0], w[1]), xytext=(w[0]-.05, w[1]-.05), fontsize=1)

    if len(false_negatives_embs) > 0:
        false_negatives_embs_proj = mapper.transform(false_negatives_embs)
        for idx, f in enumerate(false_negatives_embs_proj):
            plt.scatter(f[0], f[1], s=2, color='darkviolet')
            plt.annotate(false_negatives_l[idx] + f' - {false_negatives_scores[idx]:.4f}', xy=(f[0], f[1]), xytext=(f[0]-.05, f[1]-.05), fontsize=1)

    if len(correct_embs) > 0:
        correct_embs_proj = mapper.transform(correct_embs)
        for idx, c in enumerate(correct_embs_proj):
            plt.scatter(c[0], c[1], s=2, color='darkgreen')
            plt.annotate(correct_l[idx] + f' - {correct_scores[idx]:.4f}', xy=(c[0], c[1]), xytext=(c[0]-.05, c[1]-.05), fontsize=1)

    q_emb = reduced_embeddings[selected_query].reshape(1, -1)
    q_emb = mapper.transform(q_emb)
    plt.scatter(q_emb[:, 0], q_emb[:, 1], s=2, color='goldenrod', label='query')
    plt.title(f'Query: {selected_query}')

    plt.tight_layout()
    plt.show()

    print()

    quit()


    lookup_table = json.load(open('Dataset/task1_train_labels_2024.json', 'r'))
    queries = list(lookup_table.keys())
    embedding_dict, paragraph_count = get_embedding_dict()
    mapper = get_mapper(embedding_dict, False)

    for i in range(1): # '007846.txt'
        random_query = '098353.txt' # random.choice(queries)  # '071181''072217.txt', '097970.txt', '006704.txt', '048137.txt' '019275.txt'
        print(f'picked query: {random_query}')
        plot_umap(mapper, embedding_dict, paragraph_count, random_query, lookup_table)
    quit(0)

# os.makedirs('Dataset/translated_preprocessed_train', exist_ok=True)
# for filename in os.listdir('Dataset/regex_preprocessed_train'):
#     print(filename)
#     compare_french_english_script(Path.joinpath(Path('Dataset/regex_preprocessed_train'), Path(filename)),
#                                   Path.joinpath(Path('Dataset/translated_preprocessed_train'), Path(filename)))
