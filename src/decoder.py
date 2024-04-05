from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
path_dir = os.getcwd()

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets):
        embedded_all = torch.tanh(embedding) # [num_entity, h_dim]
        batch_size = len(triplets)
        e1_embedded = embedded_all[triplets[:, 0]].unsqueeze(1) # [num_triplets, 1, h_dim]
        e2_embedded = embedded_all[triplets[:, 2]] # [num_triplets, h_dim]
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1) # [num_triplets, 1, h_dim]
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1) # [num_triplets, 2, h_dim]
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x) # [num_triplets, channels, h_dim]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1) # [num_triplets, channels*h_dim]
        x = self.fc(x) # [num_triplets, h_dim]
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        query = F.relu(x) # [num_triplets, h_dim]
        x = torch.mm(query, embedded_all.transpose(1, 0))

        return x