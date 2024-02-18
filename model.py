from torch import nn
import torch
from transformers import AutoModel


class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False

        self.cos = torch.nn.CosineSimilarity()
        # self.linear = nn.Linear(head_size, 4)
        # self.tanh = nn.Tanh()

    def forward(self, query, evidence):
        attention_mask_q = query['attention_mask'].unsqueeze(-1)
        attention_mask_e = evidence['attention_mask'].unsqueeze(-1)
        q = self.embedder(**query).last_hidden_state
        e = self.embedder(**evidence).last_hidden_state
        q = q * attention_mask_q
        e = e * attention_mask_e

        x2 = torch.einsum('ijk,ljk->l', [q, e])

        q = q.mean(dim=1)
        e = e.mean(dim=1)
        x = self.cos(q, e)

        return x
