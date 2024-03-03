import os
import pickle
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader


EMB_IN = 1536
EMB_OUT = 3
SAMPLE_SIZE = 2
TEMPERATURE = 1.0


class EmbeddingHead(torch.nn.Module):
    def __init__(self):
        super(EmbeddingHead, self).__init__()
        self.linear1 = torch.nn.Linear(EMB_IN, 256)
        self.linear2 = torch.nn.Linear(256, EMB_OUT)
        self.relu = torch.nn.ReLU()
        self.cs_pe = torch.nn.CosineSimilarity(dim=1)
        self.cs_ne = torch.nn.CosineSimilarity(dim=2)

    def forward(self, query, pe, ne):
        # TODO: training and test inference are different
        query = self.relu(self.linear1(query))
        query = self.relu(self.linear2(query))

        pe = [self.relu(self.linear1(x)) for x in pe]
        pe = [self.relu(self.linear2(x)) for x in pe]

        ne = self.relu(self.linear1(ne))
        ne = self.relu(self.linear2(ne))

        pcs = [self.cs_pe(query[idx], x) for idx, x in enumerate(pe)]
        ncs = self.cs_ne(query.unsqueeze(1), ne)

        return pcs, ncs


class CustomDataset(Dataset):
    def __init__(self, embeddings: dict, json_dict: dict):
        self.embeddings = embeddings
        self.json_dict = json_dict

        queries_names = [key for key in json_dict if key in embeddings]
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
        negative_evidences_names = random.sample(list(sample_space), SAMPLE_SIZE)

        negative_evidences = torch.empty((0, EMB_IN))
        for e_name in negative_evidences_names:
            negative_evidences = torch.cat((negative_evidences, self.embeddings[e_name].unsqueeze(0)), dim=0)

        return query, evidence, negative_evidences


def custom_collate_fn(batch: list):
    queries = torch.stack([x[0] for x in batch])
    evidences = [x[1] for x in batch]
    negative_evidences = torch.stack([x[2] for x in batch])

    return queries, evidences, negative_evidences


def get_gpt_embeddings(json_file: str, folder_path: str, selected_files: list = None):
    files = os.listdir(folder_path)
    if 'backup' in files:
        files.remove('backup')
    embeddings = {}

    json_dict: dict[list[str]] = json.load(open(json_file))

    if selected_files:
        files = [file for file in files if file in selected_files]
        if not files:
            raise ValueError('No valid files selected')
        # Check if any selected evidence does not have a related query
        for file in selected_files:
            if file in json_dict:
                # `file` is a query
                evidences = json_dict[file]
                assert any(evidence in files for evidence in evidences), f'No evidence for query {file}'
            else:
                # `file` is an evidence
                found_query = False
                for key in json_dict:
                    if file in json_dict[key]:
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

    return embeddings, json_dict


if __name__ == '__main__':
    embeddings, json_dict = get_gpt_embeddings(json_file='Dataset/task1_train_labels_2024.json',
                                               folder_path='Dataset/gpt_embed_train')
    dataset = CustomDataset(embeddings, json_dict)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=4, shuffle=False)

    model = EmbeddingHead().to('cuda')

    for dl_row in dataloader:
        query, pe, ne = dl_row

        query = query.to('cuda')
        pe = [x.to('cuda') for x in pe]
        ne = ne.to('cuda')

        pcs, ncs = model(query, pe, ne)

    print('done')
