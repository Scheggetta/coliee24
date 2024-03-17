import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_random_seeds, average_negative_evidences, get_best_weights
from parameters import *
from dataset import TrainingDataset, QueryDataset, DocumentDataset, custom_collate_fn, get_gpt_embeddings, split_dataset


# TODO:
#  - hyperparameter grid search (parallel inference to save time?)
#  - learning to rank


class EmbeddingHead(torch.nn.Module):
    def __init__(self, hidden_units, emb_out, dropout_rate):
        super(EmbeddingHead, self).__init__()
        self.linear1 = torch.nn.Linear(EMB_IN, hidden_units)
        self.linear2 = torch.nn.Linear(hidden_units, emb_out)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, query, pe=None, ne=None, doc=None):
        if self.training:
            assert pe is not None and ne is not None, 'Positive and negative evidences must be provided during training'

            query = self.relu(self.linear1(query))
            query = self.linear2(self.dropout(query))

            pe = [self.relu(self.linear1(x)) for x in pe]
            pe = [self.linear2(self.dropout(x)) for x in pe]

            ne = [self.relu(self.linear1(x)) for x in ne]
            ne = [self.linear2(self.dropout(x)) for x in ne]

            return query, pe, ne
        else:
            assert doc is not None, 'Document must be provided during inference'
            query = self.relu(self.linear1(query))
            query = self.linear2(query)

            doc = self.relu(self.linear1(doc))
            doc = self.linear2(doc)

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
          optimizer=None,
          loss_function=None,
          cosine_loss_margin=COSINE_LOSS_MARGIN,
          lr_scheduler=None,
          lr_scheduler_mode='max',
          val_loss_function=None,
          score_function=None,
          dynamic_cutoff=DYNAMIC_CUTOFF,
          pe_weight=PE_WEIGHT,
          max_docs=MAX_DOCS,
          ratio_max_similarity=RATIO_MAX_SIMILARITY,
          pe_cutoff=PE_CUTOFF,
          metric='val_f1_score',
          save_weights=True,
          verbose=True,
          **kwargs
          ):
    if optimizer is None:
        warnings.warn('WARNING: `optimizer` of `train` function is set to None. Using Adam with default learning rate.')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if lr_scheduler is None:
        warnings.warn(
            'WARNING: `lr_scheduler` of `train` function is set to None. Using ReduceLROnPlateau with default params.')
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR, threshold=THRESHOLD,
                                                                  patience=PATIENCE, cooldown=COOLDOWN,
                                                                  mode=lr_scheduler_mode)
    if metric in ['precision', 'recall', 'val_f1_score'] and lr_scheduler_mode != 'max':
        raise ValueError('`lr_scheduler_mode` must be set to "max" when using precision, recall or f1_score as metric')
    if metric in ['train_loss', 'val_loss'] and lr_scheduler_mode != 'min':
        raise ValueError('`lr_scheduler_mode` must be set to "min" when using train_loss or val_loss as metric')
    if lr_scheduler_mode not in ['max', 'min']:
        raise ValueError('`lr_scheduler_mode` must be set to "max" or "min"')

    if loss_function is None:
        loss_function = cosine_loss_function
    if val_loss_function is None:
        val_loss_function = torch.nn.CosineEmbeddingLoss(margin=cosine_loss_margin, reduction='none')
    if score_function is None:
        score_function = torch.nn.CosineSimilarity(dim=1)

    history = {'train_loss': [], 'val_loss': [], 'val_f1_score': [], 'precision': [], 'recall': []}

    model = model.to('cuda')

    if verbose:
        pbar = tqdm(total=num_epochs, desc='Training')
    if HARD_NEGATIVE_MINING:
        starting_size = SAMPLE_SIZE

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

            if HARD_NEGATIVE_MINING:
                with torch.no_grad():
                    query_out, pe_out, ne_out = model(query, pe, ne)
                    # ne_out = ne_out.detach().to('cpu').tolist()
                    ne_filtered = []
                    for idx, ne_el in enumerate(ne_out):
                        # ne_el = torch.Tensor(ne_el).to('cuda')
                        ne_filtered.append(ne[idx][val_loss_function(query_out[idx].unsqueeze(0), ne_el,
                                                                     -torch.ones(len(ne_el)).to('cuda')) > 0])

                query_out, pe_out, ne_out = model(query, pe, ne_filtered)
            else:
                query_out, pe_out, ne_out = model(query, pe, ne)

            loss = loss_function(query_out, pe_out, ne_out, cosine_loss_margin, pe_weight)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            i += 1

        train_loss /= i

        history['train_loss'].append(train_loss)

        # Evaluate the model on the validation set
        metrics = iterate_dataset_with_model(model,
                                             validation_dataloader,
                                             val_loss_function=val_loss_function,
                                             score_function=score_function,
                                             pe_weight=pe_weight,
                                             dynamic_cutoff=dynamic_cutoff,
                                             ratio_max_similarity=ratio_max_similarity,
                                             pe_cutoff=pe_cutoff,
                                             max_docs=max_docs)
        val_loss, weighted_val_loss, pe_val_loss, ne_val_loss, precision, recall, f1_score = next(metrics)
        history['val_loss'].append(val_loss)
        history['val_f1_score'].append(f1_score)
        history['precision'].append(precision)
        history['recall'].append(recall)
        current_metric = history[metric][-1]

        lr_scheduler.step(current_metric)
        if verbose:
            description = f'Epoch {epoch + 1}/{num_epochs} - loss:{train_loss:.3f} - v_loss:{val_loss:.3f} - '
            if pe_weight is not None:
                description += f'weighted_loss:{weighted_val_loss:.3f} - pe_loss:{pe_val_loss:.3f} - ne_loss:{ne_val_loss:.3f} - '
            else:
                description += f'pe_loss:{pe_val_loss:.3f} - ne_loss:{ne_val_loss:.3f} - '
            description += f'pre:{precision:.3f} - rec:{recall:.3f} - f1:{f1_score:.3f} - lr:{lr_scheduler._last_lr[-1]:.1E}'
            pbar.set_description(description)
            pbar.update()

        if save_weights:
            if epoch == 0 or \
                    lr_scheduler_mode == 'max' and current_metric > max(history[metric][:-1]) or \
                    lr_scheduler_mode == 'min' and current_metric < min(history[metric][:-1]):
                best_weights = model.state_dict()

    if save_weights:
        model.load_state_dict(best_weights)
        if lr_scheduler_mode == 'max':
            model.save_weights(params={metric: max(history[metric])})
        else:
            model.save_weights(params={metric: min(history[metric])})

    if verbose:
        pbar.close()
    return history


def cosine_loss_function(query, pe, ne, cosine_loss_margin, pe_weight):
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


def iterate_dataset_with_model(model,
                               validation_dataloader,
                               pe_weight=None,
                               dynamic_cutoff=None,
                               ratio_max_similarity=None,
                               pe_cutoff=None,
                               max_docs=None,
                               val_loss_function=None,
                               score_function=None,
                               iterator_mode=False,
                               score_iterator_mode=False
                               ):
    if model is None:
        warnings.warn('WARNING: `model` of `iterate_dataset_with_model` function is set to None. No embedding '
                      'transformation will be done.')

        def model(query, doc):
            return query, doc
    else:
        assert isinstance(model, EmbeddingHead), 'No models with the exception of EmbeddingHead are supported.'
        model = model.to('cuda')
        model.eval()

    if val_loss_function is None:
        warnings.warn(
            'WARNING: `val_loss_function` of `iterate_dataset_with_model` function is set to None. Loss won\'t '
            'be computed.')
    if score_function is None:
        warnings.warn(
            'WARNING: `score_function` of `iterate_dataset_with_model` function is set to None. Metrics won\'t '
            'be computed.')
    if score_function is None and val_loss_function is not None:
        warnings.warn('WARNING: are you sure you want to compute loss without computing metrics?')
    if not iterator_mode and val_loss_function is None and score_function is None:
        warnings.warn(
            'WARNING: both `val_loss_function` and `score_function` of `iterate_dataset_with_model` function are '
            'set to None. No operation, except for iterating over the dataloader, will be performed.')

    if score_function is not None and (not dynamic_cutoff and pe_cutoff is None):
        raise ValueError('`pe_cutoff` must be set if `dynamic_cutoff` is set to False')
    if score_function is not None and (dynamic_cutoff and (ratio_max_similarity is None or max_docs is None)):
        raise ValueError('`ratio_max_similarity` and `max_docs` must be set if `dynamic_cutoff` is set to True')
    if score_iterator_mode and score_function is None:
        raise ValueError('`score_function` must be set if `score_iterator_mode` is set to True')
    if score_iterator_mode and iterator_mode:
        raise ValueError('`iterator_mode` and `score_iterator_mode` cannot be set to True at the same time')

    assert pe_weight is None or 0 <= pe_weight <= 1, 'Positive evidence weight must be between 0 and 1'

    q_dataloader, d_dataloader = validation_dataloader

    val_loss = torch.tensor(0.0).to('cuda')
    pe_val_loss = torch.tensor(0.0).to('cuda')
    ne_val_loss = torch.tensor(0.0).to('cuda')
    pe_count = 0
    ne_count = 0

    correctly_retrieved_cases = 0
    retrieved_cases = 0
    relevant_cases = 0

    if PREPROCESSING_DATASET_TYPE == 'test':
        pbar = tqdm(total=len(q_dataloader), desc='Testing')

    with torch.no_grad():
        for q_name, q_emb in q_dataloader:
            d_dataloader.dataset.mask(q_name[0])
            pe = list(set(d_dataloader.dataset.masked_evidences))
            pe_idxs = d_dataloader.dataset.get_indexes(pe)

            relevant_cases += len(pe)

            similarities = []

            for d_name, d_emb in d_dataloader:
                q_emb = q_emb.to('cuda')
                d_emb = d_emb.to('cuda')
                query_out, doc_out = model(q_emb, doc=d_emb)

                if iterator_mode:
                    yield query_out, doc_out

                # TODO: check if dot product is better than cosine similarity
                if score_function is not None:
                    s = score_function(query_out, doc_out)
                    for idx, el in enumerate(s):
                        similarities.append((d_name[idx], el.item()))

                if val_loss_function is not None:
                    d_idxs = d_dataloader.dataset.get_indexes(d_name)
                    targets_out = torch.Tensor([1 if x in pe_idxs else -1 for x in d_idxs]).to('cuda')

                    pe_out = doc_out[targets_out == 1]
                    pe_count += len(pe_out)
                    ne_out = doc_out[targets_out == -1]
                    ne_count += len(ne_out)

                    val_loss += val_loss_function(query_out, doc_out, targets_out).sum()
                    if len(pe_out) > 0:
                        pe_val_loss += val_loss_function(query_out, pe_out, torch.ones(len(pe_out)).to('cuda')).sum()
                    if len(ne_out) > 0:
                        ne_val_loss += val_loss_function(query_out, ne_out, -torch.ones(len(ne_out)).to('cuda')).sum()

            # F1 score
            if score_function is not None:
                similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
                if dynamic_cutoff:
                    threshold = similarities[0][1] * ratio_max_similarity
                    predicted_pe = [x for x in similarities if x[1] >= threshold]
                    predicted_pe = predicted_pe[:max_docs] if len(predicted_pe) > max_docs else predicted_pe
                else:
                    predicted_pe = similarities[:pe_cutoff]
                predicted_pe_names = [x[0] for x in predicted_pe]
                predicted_pe_idxs = d_dataloader.dataset.get_indexes(predicted_pe_names)

                if score_iterator_mode:
                    predicted_pe_scores = [x[1] for x in predicted_pe]
                    yield predicted_pe_scores

                gt = torch.zeros(len(similarities))
                gt[pe_idxs] = 1
                predictions = torch.zeros(len(similarities))
                predictions[predicted_pe_idxs] = 1

                correctly_retrieved_cases += len(gt[(gt == 1) & (predictions == 1)])
                retrieved_cases += len(predictions[predictions == 1])

            if PREPROCESSING_DATASET_TYPE == 'test':
                pbar.update()
            d_dataloader.dataset.restore()

    if score_function is not None:
        precision = correctly_retrieved_cases / retrieved_cases
        recall = correctly_retrieved_cases / relevant_cases
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    else:
        precision = -1
        recall = -1
        f1_score = -1

    if val_loss_function is not None:
        val_loss /= (pe_count + ne_count)
        pe_val_loss /= pe_count
        ne_val_loss /= ne_count
        if pe_weight is not None:
            weighted_val_loss = pe_weight * pe_val_loss + (1 - pe_weight) * ne_val_loss
        else:
            weighted_val_loss = -1
    else:
        val_loss = -1
        weighted_val_loss = -1
        pe_val_loss = -1
        ne_val_loss = -1

    if PREPROCESSING_DATASET_TYPE == 'test':
        pbar.close()
    if not iterator_mode and not score_iterator_mode:
        yield val_loss, weighted_val_loss, pe_val_loss, ne_val_loss, precision, recall, f1_score
        return


if __name__ == '__main__':
    # TODO: refactor datasets' generation (put the code in a function)
    set_random_seeds(600)
    if PREPROCESSING_DATASET_TYPE == 'train':
        json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
        train_dict, val_dict = split_dataset(json_dict, split_ratio=SPLIT_RATIO)
        print(f'Building Dataset with split ratio {SPLIT_RATIO}...')

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

        model = EmbeddingHead(hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR, threshold=THRESHOLD,
                                                                  patience=PATIENCE, cooldown=COOLDOWN)

        train(model, training_dataloader, (q_dataloader, d_dataloader), 50,
              metric='val_f1_score',
              optimizer=optimizer,
              lr_scheduler=lr_scheduler)
    else:
        json_dict = json.load(open('Dataset/task1_%s_labels_2024.json' % PREPROCESSING_DATASET_TYPE))
        test_embeddings = get_gpt_embeddings(folder_path='Dataset/gpt_embed_%s' % PREPROCESSING_DATASET_TYPE,
                                             selected_dict=json_dict)

        query_dataset = QueryDataset(test_embeddings, json_dict)
        document_dataset = DocumentDataset(test_embeddings, json_dict)
        q_dataloader = DataLoader(query_dataset, batch_size=1, shuffle=False)
        d_dataloader = DataLoader(document_dataset, batch_size=64, shuffle=False)

        model = EmbeddingHead(hidden_units=HIDDEN_UNITS, emb_out=EMB_OUT, dropout_rate=DROPOUT_RATE).to('cuda')
        model.load_weights(get_best_weights('val_f1_score'))

        val_loss_function = torch.nn.CosineEmbeddingLoss(margin=COSINE_LOSS_MARGIN, reduction='none')
        # dot product or cosine similarity?
        from utils import dot_similarity

        score_function = torch.nn.CosineSimilarity(dim=1)
        # score_function = dot_similarity

        results = iterate_dataset_with_model(model, (q_dataloader, d_dataloader),
                                             pe_weight=PE_WEIGHT,
                                             dynamic_cutoff=DYNAMIC_CUTOFF,
                                             ratio_max_similarity=RATIO_MAX_SIMILARITY,
                                             pe_cutoff=PE_CUTOFF,
                                             max_docs=MAX_DOCS,
                                             val_loss_function=val_loss_function,
                                             score_function=score_function)
        res = next(results)
        print(f'Test set results:\n val_loss: {res[0]}, weighted_val_loss: {res[1]}, pe_val_loss: {res[2]}, '
              f'ne_val_loss: {res[3]}, precision: {res[4]}, recall: {res[5]}, f1_score: {res[6]}')
    print('done')
