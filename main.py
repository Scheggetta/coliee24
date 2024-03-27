import numpy as np
from catboost import CatBoostRanker, Pool
from colorama import init as colorama_init
from colorama import Fore, Style
from pathlib import Path
import json
import pickle
import torch
from tabulate import tabulate

from catboost_ranking import convert_scores, get_metrics, get_missed_positives, Normalizer
from dataset import create_dataloaders
from embedding_head import iterate_dataset_with_model, EmbeddingHead
from parameters import *
from utils import convert_dict, compute_metrics, fancy_print, get_best_weights

from baselines.bm25_inference import BM25Custom, iterate_dataset_with_bm25, tokenize_corpus_from_dict

bm25_path = Path.joinpath(Path('Dataset'), Path('bm25_test_results_0.8.pkl'))
assert bm25_path.exists(), 'BM25 results not found. Please run the BM25 inference script first.'
test_bm25_scores = pickle.load(open(bm25_path, 'rb'))
test_bm25_scores = convert_dict(test_bm25_scores, BM25_TOP_N)
bm25_metrics = compute_metrics(test_bm25_scores)

tfidf_path = Path.joinpath(Path('Dataset'), Path('tfidf_test_results_0.8.pkl'))
assert tfidf_path.exists(), 'TFIDF results not found. Please run the TFIDF inference script first.'
test_tfidf_scores = pickle.load(open(tfidf_path, 'rb'))
test_tfidf_scores = convert_dict(test_tfidf_scores, TFIDF_TOP_N)
tfidf_metrics = compute_metrics(test_tfidf_scores)

_, qd_dataloader = create_dataloaders('test')
cs_fn = torch.nn.CosineSimilarity(dim=1)
gpt_metrics = next(iterate_dataset_with_model(model=None,
                                              validation_dataloader=qd_dataloader,
                                              score_function=cs_fn,
                                              pe_cutoff=PE_CUTOFF,
                                              verbose=True,
                                              suppress_warnings=True
                                              ))
all_ones_metrics = next(iterate_dataset_with_model(model=None,
                                                   validation_dataloader=qd_dataloader,
                                                   score_function=lambda query, doc: torch.ones((len(doc))),
                                                   pe_cutoff=PE_CUTOFF,
                                                   verbose=True,
                                                   suppress_warnings=True
                                                   ))
random_metrics = next(iterate_dataset_with_model(model=None,
                                                 validation_dataloader=qd_dataloader,
                                                 score_function=lambda query, doc: torch.rand((len(doc),)) * 2 - 1,
                                                 pe_cutoff=PE_CUTOFF,
                                                 verbose=True,
                                                 suppress_warnings=True
                                                 ))
model = EmbeddingHead(hidden_units=F1_HIDDEN_UNITS, emb_out=EMB_OUT).to('cuda')
model.load_weights(get_best_weights('val_f1_score', mode='max'))
val_loss_function = torch.nn.CosineEmbeddingLoss(margin=COSINE_LOSS_MARGIN, reduction='none')

emb_head_results = next(iterate_dataset_with_model(model, qd_dataloader,
                                                   pe_weight=PE_WEIGHT,
                                                   dynamic_cutoff=DYNAMIC_CUTOFF,
                                                   ratio_max_similarity=RATIO_MAX_SIMILARITY,
                                                   max_docs=MAX_DOCS,
                                                   val_loss_function=val_loss_function,
                                                   score_function=cs_fn,
                                                   verbose=True))

colorama_init()

method_results = [[bm25_metrics, 'Okapi BM25', Fore.MAGENTA],
                  [tfidf_metrics, 'Tf-Idf', Fore.CYAN],
                  [gpt_metrics[4:], 'GPT-Only', Fore.GREEN],
                  [all_ones_metrics[4:], 'All Ones', Fore.RED],
                  [random_metrics[4:], 'Random Inference', Fore.YELLOW],
                  [emb_head_results[4:], 'Embedding Head', Fore.BLUE]]
method_results.sort(key=lambda x: x[0][2], reverse=False)
for method in method_results:
    fancy_print(method[0], method[1], method[2])

print(f'{Style.RESET_ALL}')
print('||---------------------------------------------------||')
print('|| Final Method: Gradient Boosting on Decision Trees ||')
print('||---------------------------------------------------||\n')

model = CatBoostRanker(loss_function='YetiRank', task_type='CPU')
model.load_model('catboost_model_split.bin')
with open('Dataset/tabular_dataset_25.pkl', 'rb') as f:
    dataset = pickle.load(f)
test_group_id, test_features, test_labels, test_predicted_evidences = dataset['test']
test_pool = Pool(data=test_features, label=test_labels, group_id=test_group_id)
test_pool.set_feature_names(['f1_model', 'f1_model_dot', 'gpt', 'gpt_dot', 'bm25', 'tfidf'])
Y = model.predict(test_pool)
n_queries = len(set(test_group_id))
Y = Y.reshape(n_queries, PE_CUTOFF)
gt = np.array(test_labels).reshape(n_queries, PE_CUTOFF)
res = convert_scores(Y, gt, test_predicted_evidences)
metrics = get_metrics(res, missed_positives=get_missed_positives(test_predicted_evidences, mode='test'))

# print(f'{Fore.LIGHTGREEN_EX}Catboost: Gradient Boosting on Decision Trees')
fancy_print(metrics, 'Catboost Ranking', Fore.LIGHTGREEN_EX)
table = tabulate(model.get_feature_importance(test_pool, type='PredictionValuesChange', prettified=True),
                 headers='keys', tablefmt='heavy_outline', showindex=False)
print("Feature Importances:")
print(table)
# print(f"{Fore.LIGHTGREEN_EX}{table}{Style.RESET_ALL}")
