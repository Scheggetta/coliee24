import json
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score

from parameters import *
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings, split_dataset


seed = 243
print(f'Setting seed to {seed}')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

WHOLE_DATASET = False


if __name__ == '__main__':
    json_dict = json.load(open('Dataset/task1_train_labels_2024.json'))
    train_dict, val_dict = split_dataset(json_dict, split_ratio=0.9)

    embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train', selected_dict=json_dict)
    training_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train',
                                             selected_dict=train_dict)
    validation_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train',
                                               selected_dict=val_dict)

    # dataset = TrainingDataset(training_embeddings, train_dict)
    # training_dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=32, shuffle=False)

    if WHOLE_DATASET:
        query_dataset = QueryDataset(embeddings, json_dict)
        document_dataset = DocumentDataset(embeddings, json_dict)
    else:
        query_dataset = QueryDataset(validation_embeddings, val_dict)
        document_dataset = DocumentDataset(validation_embeddings, val_dict)
    q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    d_dataloader = DataLoader(document_dataset, batch_size=64, shuffle=False)

    cs_fn = torch.nn.CosineSimilarity(dim=1)
    loss_function = torch.nn.CosineEmbeddingLoss(reduction='none')

    pbar = tqdm(total=len(q_dataloader), desc='GPT-Only Inference')

    val_loss = 0.0
    pe_val_loss = 0.0
    ne_val_loss = 0.0
    pe_count = 0
    ne_count = 0
    f1 = 0.0
    i = 0

    correctly_retrieved_cases = 0
    retrieved_cases = 0
    relevant_cases = 0

    with torch.no_grad():
        for q_name, q_emb in q_dataloader:
            d_dataloader.dataset.mask(q_name[0])
            pe = d_dataloader.dataset.masked_evidences
            pe_idxs = d_dataloader.dataset.get_indexes(pe)

            relevant_cases += len(pe)

            similarities = []

            for d_name, d_emb in d_dataloader:
                q_emb = q_emb.to('cuda')
                d_emb = d_emb.to('cuda')

                cs = torch.rand((len(d_emb),)) * 2 - 1
                for idx, el in enumerate(cs):
                    similarities.append((d_name[idx], el.item()))

                d_idxs = d_dataloader.dataset.get_indexes(d_name)
                targets = torch.Tensor([1 if x in pe_idxs else -1 for x in d_idxs]).to('cuda')

                pe = d_emb[targets == 1]
                pe_count += len(pe)
                ne = d_emb[targets == -1]
                ne_count += len(ne)

                val_loss += loss_function(q_emb, d_emb, targets).sum()
                pe_val_loss += 0.0 if len(pe) == 0 else loss_function(q_emb, pe,
                                                                      torch.ones(len(pe)).to('cuda')).sum()
                ne_val_loss += 0.0 if len(ne) == 0 else loss_function(q_emb, ne,
                                                                      -torch.ones(len(ne)).to('cuda')).sum()

            # F1 score
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            if DYNAMIC_CUTOFF:
                threshold = similarities[0][1] * RATIO_MAX_SIMILARITY
                predicted_pe = [x for x in similarities if x[1] >= threshold]
            else:
                predicted_pe = similarities[:PE_CUTOFF]

            predicted_pe_names = [x[0] for x in predicted_pe]
            predicted_pe_idxs = d_dataloader.dataset.get_indexes(predicted_pe_names)
            gt = torch.zeros(len(similarities))
            gt[pe_idxs] = 1
            targets = torch.zeros(len(similarities))
            targets[predicted_pe_idxs] = 1

            correctly_retrieved_cases += len(gt[(gt == 1) & (targets == 1)])
            retrieved_cases += len(targets[targets == 1])
            precision = correctly_retrieved_cases / retrieved_cases
            recall = correctly_retrieved_cases / relevant_cases
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            f1 += binary_f1_score(gt, targets)

            d_dataloader.dataset.restore()

            i += 1
            pbar.set_description(f'val_loss: {val_loss / (pe_count + ne_count):.3f} - '
                                 f'pe_val_loss: {pe_val_loss / pe_count:.3f} - '
                                 f'ne_val_loss: {ne_val_loss / ne_count:.3f} - '
                                 f'f1: {f1_score:.4f}')
            pbar.update()
