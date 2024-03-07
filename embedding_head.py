import os
import json
import random
from math import ceil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score
from tqdm import tqdm

from parameters import *
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings


# TODO:
#  - focal loss
#  - hyperparameter grid search
#  - alternative model architecture


seed = 62
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class EmbeddingHead(torch.nn.Module):
    def __init__(self):
        super(EmbeddingHead, self).__init__()
        self.linear1 = torch.nn.Linear(EMB_IN, 256)
        self.linear2 = torch.nn.Linear(256, EMB_OUT)
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

    def save_weights(self):
        os.makedirs('Checkpoints', exist_ok=True)
        now = datetime.now()
        file_path = Path.joinpath(Path('Checkpoints'), Path(f'weights_{now.strftime("%d_%m_%H-%M-%S")}.pt'))
        torch.save(self.state_dict(), file_path)

    def load_weights(self, file_path: Path):
        assert file_path.exists(), f'No weights found in {str(file_path)}'
        self.load_state_dict(torch.load(file_path))





def train(model, train_dataloader, validation_dataloader, num_epochs, save_weights=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CosineEmbeddingLoss(reduction='none')
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ceil(num_epochs / 5), gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.05, total_iters=num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, threshold=1e-3, patience=1,
                                                              cooldown=3)

    history = {'train_loss': [], 'val_loss': [], 'val_f1_score': []}

    model = model.to('cuda')

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

            loss = compute_loss(query_out, pe_out, ne_out, loss_function, PE_WEIGHT)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            i += 1

        # lr_scheduler.step()

        train_loss /= i
        history['train_loss'].append(train_loss)

        # Evaluate the model on the validation set
        v_losses = evaluate_model(model, validation_dataloader, pe_weight=PE_WEIGHT, dynamic_cutoff=DYNAMIC_CUTOFF)
        val_loss = v_losses[1] if v_losses[1] else v_losses[0]
        history['val_loss'].append(val_loss)
        val_f1 = v_losses[-1]
        history['val_f1_score'].append(val_f1)

        lr_scheduler.step(val_loss)
        pbar.set_description(
            f'Epoch {epoch + 1}/{num_epochs} - loss: {train_loss:.3f} - val_loss: {val_loss:.3f} - val_f1: {val_f1:.3f} - lr: {lr_scheduler._last_lr[-1]:.6f}')
        pbar.update()

        if save_weights:
            if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
                # print(f'New best model found (val_loss = {val_loss})! Saving it...')
                best_weights = model.state_dict()

    if save_weights:
        model.load_state_dict(best_weights)
        model.save_weights()

    pbar.close()
    print('Finished Training\n')
    return history


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


def evaluate_model(model, validation_dataloader, pe_weight=None, dynamic_cutoff=False):
    assert pe_weight is None or 0 <= pe_weight <= 1, 'Positive evidence weight must be between 0 and 1'
    model.eval()

    q_dataloader, d_dataloader = validation_dataloader
    loss_function = torch.nn.CosineEmbeddingLoss(reduction='none')

    val_loss = 0.0
    pe_val_loss = 0.0
    ne_val_loss = 0.0
    pe_count = 0
    ne_count = 0
    f1 = 0.0

    with torch.no_grad():
        for q_name, q_emb in q_dataloader:
            d_dataloader.dataset.mask(q_name[0])
            pe = d_dataloader.dataset.masked_evidences
            pe_idxs = d_dataloader.dataset.get_indexes(pe)

            similarities = []

            for d_name, d_emb in d_dataloader:
                q_emb = q_emb.to('cuda')
                d_emb = d_emb.to('cuda')
                query_out, doc_out = model(q_emb, doc=d_emb)

                cs = model.cs(query_out, doc_out)
                for idx, el in enumerate(cs):
                    similarities.append((d_name[idx], el.item()))

                d_idxs = d_dataloader.dataset.get_indexes(d_name)
                targets = torch.Tensor([1 if x in pe_idxs else -1 for x in d_idxs]).to('cuda')

                pe = doc_out[targets == 1]
                pe_count += len(pe)
                ne = doc_out[targets == -1]
                ne_count += len(ne)

                val_loss += loss_function(query_out, doc_out, targets).sum()
                pe_val_loss += 0.0 if len(pe) == 0 else loss_function(query_out, pe, torch.ones(len(pe)).to('cuda')).sum()
                ne_val_loss += 0.0 if len(ne) == 0 else loss_function(query_out, ne, -torch.ones(len(ne)).to('cuda')).sum()

            # F1 score
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            if dynamic_cutoff:
                threshold = similarities[0][1] * RATIO_MAX_SIMILARITY
                predicted_pe = [x for x in similarities if x[1] >= threshold]
            else:
                predicted_pe = similarities[:PE_CUTOFF]

            # predicted_pe = similarities[:PE_CUTOFF]
            # threshold = similarities[0][1] * RATIO_MAX_SIMILARITY
            # predicted_pe = [x for x in similarities if x[1] >= threshold]
            predicted_pe_names = [x[0] for x in predicted_pe]
            predicted_pe_idxs = d_dataloader.dataset.get_indexes(predicted_pe_names)
            gt = torch.zeros(len(similarities))
            gt[pe_idxs] = 1
            targets = torch.zeros(len(similarities))
            targets[predicted_pe_idxs] = 1
            f1 += binary_f1_score(gt, targets)

            d_dataloader.dataset.restore()

    val_loss /= (pe_count + ne_count)
    pe_val_loss /= pe_count
    ne_val_loss /= ne_count
    f1 /= len(q_dataloader)
    if pe_weight is not None:
        weighted_val_loss = pe_weight * pe_val_loss + (1 - pe_weight) * ne_val_loss
    else:
        weighted_val_loss = None

    return val_loss, weighted_val_loss, pe_val_loss, ne_val_loss, f1


def split_dataset(json_dict, split_ratio=0.9):
    keys = list(json_dict.keys())
    random.shuffle(keys)

    train_size = ceil(len(json_dict) * split_ratio)
    train_dict = {key: json_dict[key] for key in keys[:train_size]}
    val_dict = {key: json_dict[key] for key in keys[train_size:]}

    return train_dict, val_dict


def predict(model, q_dataloader, d_dataloader):
    model.eval()
    results = {}
    GT = {}
    for q_name, q_emb in q_dataloader:
        d_dataloader.dataset.mask(q_name[0])
        pe = d_dataloader.dataset.masked_evidences
        pe_idxs = d_dataloader.dataset.get_indexes(pe)
        q_emb = q_emb.to('cuda')

        results[q_name] = []
        GT[q_name] = []
        for d_name, d_emb in d_dataloader:
            d_emb = d_emb.to('cuda')
            d_idxs = d_dataloader.dataset.get_indexes(d_name)
            query_out, doc_out = model(q_emb, doc=d_emb)

            similarities = [model.cs(query_out, x.unsqueeze(0)) for x in doc_out]
            results[q_name] += [1 if similarities[n] > 0 else 0 for n in range(len(similarities))]
            GT[q_name] += torch.Tensor([1 if x in pe_idxs else -1 for x in d_idxs])

        d_dataloader.dataset.restore()

    return results, GT


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

    train(model, training_dataloader, (q_dataloader, d_dataloader), 30)

    # model.load_weights(Path('Checkpoints/weights_06_03_18-15-27.pt'))
    # res, GT = predict(model, q_dataloader, d_dataloader)
    # sample = list(res.keys())[0]
    # print(res[sample])
    # print(GT[sample])

    print('done')
