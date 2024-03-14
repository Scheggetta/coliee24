import os
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score
from tqdm import tqdm

from utils import set_random_seeds
from parameters import *
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings, split_dataset

# TODO:
#  - hyperparameter grid search (parallel inference to save time?)
#  - learning to rank
#  - check that repeated evidences don't influence loss and evaluation functions and f1 score


class EmbeddingHead(torch.nn.Module):
    def __init__(self, hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT):
        super(EmbeddingHead, self).__init__()
        self.linear1 = torch.nn.Linear(EMB_IN, HIDDEN_UNITS)
        self.linear2 = torch.nn.Linear(HIDDEN_UNITS, EMB_OUT)
        self.relu = torch.nn.ReLU()
        self.cs = torch.nn.CosineSimilarity(dim=1)
        # self.cs = torch.nn.CosineSimilarity(dim=2)

    def forward(self, query, pe=None, ne=None, doc=None):
        if self.training:
            assert pe is not None and ne is not None, 'Positive and negative evidences must be provided during training'

            query = self.relu(self.linear1(query))
            query = self.linear2(query)

            pe = [self.relu(self.linear1(x)) for x in pe]
            pe = [self.linear2(x) for x in pe]

            ne = self.relu(self.linear1(ne))
            ne = self.linear2(ne)

            # pcs = [self.cs_pe(query[idx], x) for idx, x in enumerate(pe)]
            # ncs = self.cs_ne(query.unsqueeze(1), ne)

            return query, pe, ne
        else:
            assert doc is not None, 'Document must be provided during inference'
            query = self.relu(self.linear1(query))
            query = self.linear2(query)

            doc = self.relu(self.linear1(doc))
            doc = self.linear2(doc)

            # cs = self.cs_pe(query, doc)

            return query, doc

    def save_weights(self, params=None):
        if params is None:
            params = {}
        os.makedirs('Checkpoints', exist_ok=True)
        now = datetime.now()
        name = f'weights_{now.strftime("%d_%m_%H-%M-%S")}'
        for par in params:
            name += f'_{par}_{params[par]}'
        name += '.pt'
        file_path = Path.joinpath(Path('Checkpoints'), Path(name))
        torch.save(self.state_dict(), file_path)

    def load_weights(self, file_path: Path):
        assert file_path.exists(), f'No weights found in {str(file_path)}'
        self.load_state_dict(torch.load(file_path))


def train(model,
          train_dataloader,
          validation_dataloader,
          num_epochs=30,
          save_weights=True,
          optimizer=None,
          loss_function=None,
          cosine_loss_margin=COSINE_LOSS_MARGIN,
          lr_scheduler=None,
          pe_weight=PE_WEIGHT,
          max_docs=MAX_DOCS,
          ratio_max_similarity=RATIO_MAX_SIMILARITY,
          pe_cutoff=PE_CUTOFF,
          metric='val_f1_score',
          verbose=True
          ):

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR, threshold=THRESHOLD,
                                                                  patience=PATIENCE, cooldown=COOLDOWN, mode='min')
        if metric in ['precision', 'recall', 'val_f1_score']:
            lr_scheduler.mode = 'max'
    if loss_function is None:
        loss_function = cosine_loss_function

    history = {'train_loss': [], 'val_loss': [], 'val_f1_score': [], 'precision': [], 'recall': []}

    model = model.to('cuda')

    if verbose:
        pbar = tqdm(total=num_epochs, desc='Training')

    best_weights = model.state_dict()
    for epoch in range(num_epochs):
        model.train(True)

        train_loss = 0.0
        i = 0
        for dl_row in train_dataloader:
            query, pe, ne = dl_row

            query = query.to('cuda')
            pe = [x.to('cuda') for x in pe]
            ne = ne.to('cuda')

            optimizer.zero_grad()

            query_out, pe_out, ne_out = model(query, pe, ne)

            loss = loss_function(query_out, pe_out, ne_out, cosine_loss_margin, pe_weight)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            i += 1

        train_loss /= i
        history['train_loss'].append(train_loss)

        # Evaluate the model on the validation set
        metrics = evaluate_model(model, validation_dataloader, pe_weight=pe_weight, dynamic_cutoff=DYNAMIC_CUTOFF,
                                 ratio_max_similarity=ratio_max_similarity, pe_cutoff=pe_cutoff, max_docs=max_docs)
        val_loss, weighted_val_loss, pe_val_loss, ne_val_loss, precision, recall, f1_score = metrics
        history['val_loss'].append(val_loss)
        history['val_f1_score'].append(f1_score)
        history['precision'].append(precision)
        history['recall'].append(recall)
        current_metric = history[metric][-1]

        lr_scheduler.step(current_metric)
        if verbose:
            description = f'Epoch {epoch + 1}/{num_epochs} - loss:{train_loss:.3f} - v_loss:{val_loss:.3f} - '
            if weighted_val_loss is not None:
                description += f'weighted_loss:{weighted_val_loss:.3f} - pe_loss:{pe_val_loss:.3f} - ne_loss:{ne_val_loss:.3f} - '
            description += f'pre:{precision:.3f} - rec:{recall:.3f} - f1:{f1_score:.3f} - lr:{lr_scheduler._last_lr[-1]:.1E}'
            pbar.set_description(description)
            pbar.update()

        if save_weights:
            if epoch == 0 or current_metric > max(history[metric][:-1]):
                # print(f'New best model found (val_loss = {val_loss})! Saving it...')
                best_weights = model.state_dict()

    if save_weights:
        model.load_state_dict(best_weights)
        model.save_weights(params={metric: max(history[metric])})

    if verbose:
        pbar.close()
        print('Finished Training\n')
    return history


def cosine_loss_function(query, pe, ne, cosine_loss_margin, pe_weight=None):
    assert pe_weight is None or 0 <= pe_weight <= 1, 'Positive evidence weight must be between 0 and 1'
    loss_function = torch.nn.CosineEmbeddingLoss(margin=cosine_loss_margin, reduction='none')

    loss = torch.tensor(0.0).to('cuda')
    for idx, query_el in enumerate(query):
        query_el = query_el.unsqueeze(0)
        pe_el = pe[idx]
        ne_el = ne[idx]

        pe_loss = loss_function(query_el, pe_el, torch.ones(1).to('cuda'))
        ne_loss = loss_function(query_el, ne_el, -torch.ones(1).to('cuda'))

        if pe_weight is None:
            losses = torch.cat((pe_loss, ne_loss))
            loss += losses.mean()
        else:
            pe_loss = pe_loss.mean()
            ne_loss = ne_loss.mean()
            loss += pe_weight * pe_loss + (1 - pe_weight) * ne_loss

    return loss / len(query)


def evaluate_model(model, validation_dataloader, pe_weight=None, dynamic_cutoff=False,
                   ratio_max_similarity=RATIO_MAX_SIMILARITY, pe_cutoff=PE_CUTOFF, max_docs=MAX_DOCS):
    assert pe_weight is None or 0 <= pe_weight <= 1, 'Positive evidence weight must be between 0 and 1'
    model.eval()

    q_dataloader, d_dataloader = validation_dataloader
    loss_function = torch.nn.CosineEmbeddingLoss(reduction='none')
    # loss_function = contrastive_loss_function
    # inner_loss_fn = torch.nn.CrossEntropyLoss()

    val_loss = torch.tensor(0.0).to('cuda')
    pe_val_loss = torch.tensor(0.0).to('cuda')
    ne_val_loss = torch.tensor(0.0).to('cuda')
    pe_count = 0
    ne_count = 0

    correctly_retrieved_cases = 0
    retrieved_cases = 0
    relevant_cases = 0

    with torch.no_grad():
        for q_name, q_emb in q_dataloader:
            d_dataloader.dataset.mask(q_name[0])
            pe = list(set(d_dataloader.dataset.masked_evidences))
            pe_idxs = d_dataloader.dataset.get_indexes(pe)

            relevant_cases += len(pe)

            similarities = []
            positive_contributions = torch.Tensor([]).to('cuda')
            negative_contribution = torch.Tensor([]).to('cuda')

            for d_name, d_emb in d_dataloader:
                q_emb = q_emb.to('cuda')
                d_emb = d_emb.to('cuda')
                query_out, doc_out = model(q_emb, doc=d_emb)

                cs = model.cs(query_out, doc_out)
                for idx, el in enumerate(cs):
                    similarities.append((d_name[idx], el.item()))

                d_idxs = d_dataloader.dataset.get_indexes(d_name)
                targets = torch.Tensor([1 if x in pe_idxs else -1 for x in d_idxs]).to('cuda')

                pe_out = doc_out[targets == 1]
                pe_count += len(pe_out)
                ne_out = doc_out[targets == -1]
                ne_count += len(ne_out)

                # negative_contribution += torch.sum(torch.exp(model.cs(query_out, ne_out)))

                # negative_contribution = torch.cat((negative_contribution, torch.einsum('ij,ij->i', query_out, ne_out)))
                # for el in pe_out:
                #     pc = torch.einsum('ij,j->', query_out, el)
                #     positive_contributions = torch.cat((positive_contributions, pc.unsqueeze(0)))

                # val_loss += loss_function(model.cs, query_out, pe_out, ne_out)

                val_loss += loss_function(query_out, doc_out, targets).sum()
                if len(pe_out) > 0:
                    pe_val_loss += loss_function(query_out, pe_out, torch.ones(len(pe_out)).to('cuda')).sum()
                if len(ne_out) > 0:
                    ne_val_loss += loss_function(query_out, ne_out, -torch.ones(len(ne_out)).to('cuda')).sum()

            # assert len(positive_contributions) == len(pe), 'Positive contributions must be equal to the number of positive evidences'
            # loss_score = torch.tensor(0.0).to('cuda')
            # loss_targets = torch.zeros(1 + len(negative_contribution)).to('cuda')
            # loss_targets[0] = 1
            # for positive in positive_contributions:
            #     logits = torch.cat((positive.unsqueeze(0), negative_contribution))
            #     loss_score += inner_loss_fn(logits, loss_targets)
            # loss_score /= len(positive_contributions)
            # val_loss += loss_score

            # F1 score
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            if dynamic_cutoff:
                threshold = similarities[0][1] * ratio_max_similarity
                predicted_pe = [x for x in similarities if x[1] >= threshold]
                predicted_pe = predicted_pe[:max_docs] if len(predicted_pe) > max_docs else predicted_pe
            else:
                predicted_pe = similarities[:pe_cutoff]

            predicted_pe_names = [x[0] for x in predicted_pe]
            predicted_pe_idxs = d_dataloader.dataset.get_indexes(predicted_pe_names)
            gt = torch.zeros(len(similarities))
            gt[pe_idxs] = 1
            targets = torch.zeros(len(similarities))
            targets[predicted_pe_idxs] = 1

            correctly_retrieved_cases += len(gt[(gt == 1) & (targets == 1)])
            retrieved_cases += len(targets[targets == 1])

            d_dataloader.dataset.restore()

    precision = correctly_retrieved_cases / retrieved_cases
    recall = correctly_retrieved_cases / relevant_cases
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # val_loss /= len(q_dataloader)
    val_loss /= (pe_count + ne_count)
    pe_val_loss /= pe_count
    ne_val_loss /= ne_count
    if pe_weight is not None:
        weighted_val_loss = pe_weight * pe_val_loss + (1 - pe_weight) * ne_val_loss
    else:
        weighted_val_loss = None

    return val_loss, weighted_val_loss, pe_val_loss, ne_val_loss, precision, recall, f1_score
    # return val_loss, None, pe_val_loss, ne_val_loss, precision, recall, f1_score


def get_best_weights():
    weights = [x for x in os.listdir('Checkpoints') if x.endswith('.pt')]
    best_path = sorted(weights, key=lambda x: float(x.split(sep='_')[-1][:-3]))[-1]
    return Path.joinpath(Path('Checkpoints'), Path(best_path))


if __name__ == '__main__':
    set_random_seeds(600)
    if PREPROCESSING_DATASET_TYPE == 'train':
        json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
        split_ratio = 0.9
        train_dict, val_dict = split_dataset(json_dict, split_ratio=split_ratio)
        print(f'Building Dataset with split ratio {split_ratio}...')

        training_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_%s' % PREPROCESSING_DATASET_TYPE,
                                                 selected_dict=train_dict)
        validation_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_%s' % PREPROCESSING_DATASET_TYPE,
                                                   selected_dict=val_dict)

        dataset = TrainingDataset(training_embeddings, train_dict)
        training_dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=32, shuffle=False)

        query_dataset = QueryDataset(validation_embeddings, val_dict)
        document_dataset = DocumentDataset(validation_embeddings, val_dict)
        q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
        d_dataloader = DataLoader(document_dataset, batch_size=128, shuffle=False)

        model = EmbeddingHead().to('cuda')
        # train(model, training_dataloader, (q_dataloader, d_dataloader), 30, metric='val_loss')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR, threshold=THRESHOLD,
                                                                  patience=PATIENCE, cooldown=COOLDOWN)
        train(model, training_dataloader, (q_dataloader, d_dataloader), 30,
              metric='val_f1_score',
              optimizer=optimizer,
              loss_function=contrastive_loss_function,
              lr_scheduler=lr_scheduler)
    else:
        json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
        test_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_%s' % PREPROCESSING_DATASET_TYPE,
                                             selected_dict=json_dict)

        query_dataset = QueryDataset(test_embeddings, json_dict)
        document_dataset = DocumentDataset(test_embeddings, json_dict)
        q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
        d_dataloader = DataLoader(document_dataset, batch_size=64, shuffle=False)

        model = EmbeddingHead().to('cuda')
        # model.load_weights(Path(get_best_weights()))
        model.load_weights(Path('Checkpoints/weights_14_03_12-48-37_val_loss_0.2682577073574066.pt'))
        print('Beginning test procedure...')
        results = evaluate_model(model, (q_dataloader, d_dataloader),
                                 pe_weight=PE_WEIGHT,
                                 dynamic_cutoff=DYNAMIC_CUTOFF,
                                 ratio_max_similarity=RATIO_MAX_SIMILARITY,
                                 pe_cutoff=PE_CUTOFF,
                                 max_docs=MAX_DOCS)
        print(f'Test set: val_f1: {results}')

    print('done')
