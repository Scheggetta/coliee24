from torch import nn
import torch
from transformers import AutoModel, BertModel, RobertaModel


class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.embedder = AutoModel.from_pretrained(name)
        for param in self.embedder.parameters():
            param.requires_grad = False
        # self.linear = nn.Linear(head_size, 4)
        # self.tanh = nn.Tanh()

    def forward(self, query, evidence):
        # attention_mask = x['attention_mask'].unsqueeze(-1)
        q = self.embedder(**query).last_hidden_state
        e = self.embedder(**evidence).last_hidden_state
        # x = x * attention_mask
        # x = x.mean(dim=1)
        # x = self.linear(x)
        # x = (self.tanh(x) + 1) / 2
        # Dot product
        x = torch.matmul(q, e)
        return x
