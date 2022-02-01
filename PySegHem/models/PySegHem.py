import torch

import torch.nn as nn
from Cnn_Encoder import Cnn_Encoder
from GCN import GCN_Layer
from SCG import SCG


class PySegHem(nn.Module):
    def __init__(self):
        super(PySegHem, self).__init__()
        self.Cnn = Cnn_Encoder()
        self.scg = SCG(node_size=(128, 128))
        self.gnn1 = GCN_Layer(128, 60, bnorm=False, dropout=0.5)
        self.gnn2 = GCN_Layer(60, 8, bnorm=False, dropout=0.5)
        self.seq1 = self._create_block(1, 16, dropout_prob=0.3)
        self.seq2 = self._create_block(32, 16, dropout_prob=0.1)
        self.seq3 = self._create_block(32, 16, dropout_prob=0.1)
        self.seq4 = self._create_block(32, 16, dropout_prob=0)
        self.final_seq = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1),
                                       nn.Sigmoid())


    def forward(self,x):
        x = self.Cnn(x)
        adj_matrix, feature_matrix, loss_kl, z_hat = self.scg(x['output'])
        b, h, w = feature_matrix.size()
        gnn_output = self.gnn1((feature_matrix, adj_matrix))
        gnn_output = self.gnn2(gnn_output)
        graph_embeddings = gnn_output[0]


        graph_embeddings = graph_embeddings.view(b, 1, 32, 32)
        x_after_seq1 = self.seq1(graph_embeddings)
        x_after_cat = torch.cat([x['skip_4'], x_after_seq1],dim=1)
        x_after_seq2 = self.seq2(x_after_cat)
        x_after_cat = torch.cat([x['skip_3'], x_after_seq2],dim=1)
        x_after_seq3 = self.seq3(x_after_cat)
        x_after_cat = torch.cat([x['skip_2'], x_after_seq3],dim=1)
        x_after_seq4 = self.seq4(x_after_cat)
        x_after_cat = torch.cat([x['skip_1'], x_after_seq4],dim=1)
        activated_logits = self.final_seq(x_after_cat)
        return activated_logits



    @classmethod
    def _create_block(cls, input_channels: int, output_channels: int, dropout_prob: float):
        return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                             nn.Dropout2d(dropout_prob),
                             nn.BatchNorm2d(output_channels),
                             nn.ReLU(),
                             nn.Upsample(scale_factor=2),
                             )





