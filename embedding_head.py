import os
import pickle
import json
import random
import copy
from math import ceil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from setlist import SetList


seed = 62
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


EMB_IN = 1536
EMB_OUT = 50
SAMPLE_SIZE = 15
TEMPERATURE = 1.0


class EmbeddingHead(torch.nn.Module):
    def __init__(self):
        super(EmbeddingHead, self).__init__()
        self.linear1 = torch.nn.Linear(EMB_IN, 256)
        self.linear2 = torch.nn.Linear(256, EMB_OUT)
        self.relu = torch.nn.ReLU()
        # self.cs_pe = torch.nn.CosineSimilarity(dim=1)
        # self.cs_ne = torch.nn.CosineSimilarity(dim=2)

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


class TrainingDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings = embeddings
        self.json_dict = json_dict

        queries_names = [key for key in json_dict if key in embeddings]
        # TODO: fix the reproducibility issue of set
        self.all_evidences = set([evidence for query_name in queries_names for evidence in json_dict[query_name]])

        self.queries = []
        for q_name in queries_names:
            self.queries.append((q_name, embeddings[q_name]))

        self.evidences = []
        for q_name in queries_names:
            single_evidences = torch.empty((0, EMB_IN))
            evidences = json_dict[q_name]
            for e_name in evidences:
                single_evidences = torch.cat((single_evidences, embeddings[e_name].unsqueeze(0)), dim=0)
            self.evidences.append((evidences, single_evidences))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query_name = self.queries[index][0]
        query = self.queries[index][1]
        evidence_names = self.evidences[index][0]
        evidence = self.evidences[index][1]

        excluded_documents = evidence_names.copy()
        excluded_documents.append(query_name)
        excluded_documents = set(excluded_documents)

        sample_space = self.all_evidences - excluded_documents

        # negative_evidences_names = []
        # for i in range(SAMPLE_SIZE):
        #     idx = torch.randint(0, len(sample_space), (1,)).item()
        #     el = ...
        negative_evidences_names = random.sample(list(sample_space), SAMPLE_SIZE)

        negative_evidences = torch.empty((0, EMB_IN))
        for e_name in negative_evidences_names:
            negative_evidences = torch.cat((negative_evidences, self.embeddings[e_name].unsqueeze(0)), dim=0)

        return query, evidence, negative_evidences


class QueryDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings = embeddings
        self.json_dict = json_dict

        queries_names = [key for key in json_dict if key in embeddings]
        self.queries = [(q_name, embeddings[q_name]) for q_name in queries_names]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]


class DocumentDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings_dict = embeddings
        self.json_dict = json_dict

        self.embeddings = []
        self.masked_query = None
        self.masked_evidences = None

        for key in embeddings:
            self.embeddings.append((key, embeddings[key]))

    def __len__(self):
        return len(self.embeddings)

    def mask(self, query_name):
        if self.masked_query:
            raise ValueError('A document is already masked')
        q_embedding = self.embeddings_dict[query_name]
        self.masked_evidences = self.json_dict[query_name]

        self.masked_query = (query_name, q_embedding)
        self.embeddings.remove((query_name, q_embedding))

    def restore(self):
        if self.masked_query:
            self.embeddings.append(self.masked_query)
            self.masked_query = None
            self.masked_evidences = None
        else:
            raise ValueError('No element to restore')

    def __getitem__(self, index):
        return self.embeddings[index]

    def get_indexes(self, filenames):
        return [self.embeddings.index((x, self.embeddings_dict[x])) for x in filenames]


def custom_collate_fn(batch: list):
    queries = torch.stack([x[0] for x in batch])
    evidences = [x[1] for x in batch]
    negative_evidences = torch.stack([x[2] for x in batch])

    return queries, evidences, negative_evidences


def get_gpt_embeddings(folder_path: str, selected_dict: dict):
    embeddings = {}

    files = []
    for key, values in selected_dict.items():
        files.append(key)
        files.extend(values)
    files = SetList(files)

    # Check if any selected evidence does not have a related query
    for file in files:
        if file in selected_dict:
            # `file` is a query
            evidences = selected_dict[file]
            assert any(evidence in files for evidence in evidences), f'No evidence for query {file}'
        else:
            # `file` is an evidence
            found_query = False
            for key in selected_dict:
                if file in selected_dict[key]:
                    found_query = True
                    break
            assert found_query, f'No query for evidence {file}'

    for file in files:
        with open(os.path.join(folder_path, file), 'rb') as f:
            e = pickle.load(f)
            n_paragraphs = len(e) // EMB_IN
            assert len(e) % EMB_IN == 0

            e = torch.tensor(e).view(n_paragraphs, EMB_IN).mean(dim=0)
            embeddings[file] = e

    return embeddings


def train(model, train_dataloader, validation_dataloader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CosineEmbeddingLoss(reduction='none')
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ceil(num_epochs / 5), gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.05, total_iters=num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=1e-3, patience=1,
                                                              cooldown=1)

    history = {'train_loss': [], 'val_loss': []}

    model = model.to('cuda')
    best_model = model

    pbar = tqdm(total=num_epochs, desc='Training')

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

            loss = compute_loss(query_out, pe_out, ne_out, loss_function)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            i += 1

        # lr_scheduler.step()

        train_loss /= i
        lr_scheduler.step(train_loss)
        history['train_loss'].append(train_loss)

        # Evaluate the model on the validation set
        vloss = evaluate_model(model, validation_dataloader)
        history['val_loss'].append(vloss)

        pbar.set_description(
            f'Epoch {epoch + 1}/{num_epochs} - loss: {train_loss:.3f} - val_loss: {vloss[0]:.3f} - lr: {lr_scheduler._last_lr[-1]:.6f}')
        pbar.update()

        # if macro_f1_score > history['best_val_macro_f1']:
        #     print('New best model found! Saving it...')
        #     history['best_val_macro_f1'] = macro_f1_score
        #     best_model = copy.deepcopy(model)

    pbar.close()
    print('Finished Training\n')
    return best_model, history


def compute_loss(query, pe, ne, loss_function, pe_weight=None):
    # TODO: focal loss
    assert pe_weight is None or 0 <= pe_weight <= 1, 'Positive evidence weight must be between 0 and 1'

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


def evaluate_model(model, validation_dataloader, pe_weight=None):
    assert pe_weight is None or 0 <= pe_weight <= 1, 'Positive evidence weight must be between 0 and 1'
    model.eval()

    q_dataloader, d_dataloader = validation_dataloader
    loss_function = torch.nn.CosineEmbeddingLoss(reduction='none')

    val_loss = 0.0
    pe_val_loss = 0.0
    ne_val_loss = 0.0
    i = 0
    with torch.no_grad():
        for q_name, q_emb in q_dataloader:
            d_dataloader.dataset.mask(q_name[0])
            pe = d_dataloader.dataset.masked_evidences
            pe_idxs = d_dataloader.dataset.get_indexes(pe)

            for d_name, d_emb in d_dataloader:
                q_emb = q_emb.to('cuda')
                d_emb = d_emb.to('cuda')
                query_out, doc_out = model(q_emb, doc=d_emb)

                d_idxs = d_dataloader.dataset.get_indexes(d_name)
                targets = torch.Tensor([1 if x in pe_idxs else -1 for x in d_idxs]).to('cuda')

                pe = doc_out[targets == 1]
                ne = doc_out[targets == -1]

                val_loss += loss_function(query_out, doc_out, targets).mean()
                pe_val_loss += 0.0 if len(pe) == 0 else loss_function(query_out, pe, torch.ones(len(pe)).to('cuda')).mean()
                ne_val_loss += 0.0 if len(ne) == 0 else loss_function(query_out, ne, -torch.ones(len(ne)).to('cuda')).mean()
                i += 1

            d_dataloader.dataset.restore()

    val_loss /= i
    pe_val_loss /= i
    ne_val_loss /= i
    if pe_weight is not None:
        weighted_val_loss = pe_weight * pe_val_loss + (1 - pe_weight) * ne_val_loss
    else:
        weighted_val_loss = None

    return val_loss, weighted_val_loss, pe_val_loss, ne_val_loss


def split_dataset(json_dict, split_ratio=0.9):
    keys = list(json_dict.keys())
    random.shuffle(keys)

    train_size = ceil(len(json_dict) * split_ratio)
    train_dict = {key: json_dict[key] for key in keys[:train_size]}
    val_dict = {key: json_dict[key] for key in keys[train_size:]}

    return train_dict, val_dict


if __name__ == '__main__':
    json_dict = json.load(open('Dataset/task1_train_labels_2024.json'))
    train_dict, val_dict = split_dataset(json_dict, split_ratio=0.9)

    training_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train',
                                             selected_dict=train_dict)
    validation_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_train',
                                               selected_dict=val_dict)

    dataset = TrainingDataset(training_embeddings, train_dict)
    training_dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=32, shuffle=False)

    query_dataset = QueryDataset(validation_embeddings, val_dict)
    document_dataset = DocumentDataset(validation_embeddings, val_dict)
    q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
    d_dataloader = DataLoader(document_dataset, batch_size=64, shuffle=False)

    model = EmbeddingHead().to('cuda')

    train(model, training_dataloader, (q_dataloader, d_dataloader), 20)

    print('done')
