import torch
from torch import Tensor
import torch.nn as nn


class SCG(nn.Module):
    def __init__(self, node_size: int = (64, 64), add_diag: bool = True,
                 dropout: float = 0.6):
        super(SCG, self).__init__()
        self.in_ch = 1
        self.node_size = node_size
        self.hidden_ch = 1
        self.nodes = node_size[0] * node_size[1]
        self.add_diag = add_diag
        self.pool = nn.AdaptiveAvgPool2d(node_size)
        self.dropout = dropout

        self.mu = nn.Sequential(
            nn.Conv2d(self.in_ch, self.hidden_ch, 3, padding=1, bias=True),
            nn.Dropout(self.dropout),
        )

        self.logvar = nn.Sequential(
            nn.Conv2d(self.in_ch, self.hidden_ch, 1, 1, bias=True),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        B, C, H, W = x.size()

        gx = self.pool(x)

        mu, log_var = self.mu(gx), self.logvar(gx)

        std = torch.exp(log_var)

        if self.training:

            eps = torch.randn_like(std)
            z = mu + std * eps

        else:
            z = mu + std

        z = z.squeeze(1)
        A = torch.bmm(z, z.permute(0, 2, 1))
        A = torch.relu(A)

        Ad = torch.diagonal(A, dim1=1, dim2=2)
        mean = torch.mean(Ad, dim=1)
        gama = torch.sqrt(1 + 1.0 / mean).unsqueeze(-1).unsqueeze(-1)

        dl_loss = gama.mean() * torch.log(Ad[Ad < 1] + 1.e-7).sum() / (A.size(0) * A.size(1) * A.size(2))

        kl_loss = -0.5 / self.nodes * torch.mean(
            torch.sum(1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1)
        )

        loss = kl_loss - dl_loss

        if self.add_diag:
            diag = []
            for i in range(Ad.shape[0]):
                diag.append(torch.diag(Ad[i, :]).unsqueeze(0))

            A = A + gama * torch.cat(diag, 0)



        A = self.laplacian_matrix(A, self_loop=True)

        z_hat = gama.mean() * mu.reshape(B, self.nodes, self.hidden_ch) * (1. - log_var.reshape(B, self.nodes, self.hidden_ch))

        return A, gx.squeeze(1), loss, z_hat

    @classmethod
    def laplacian_matrix(cls, A, self_loop=False):
        '''
        Computes normalized Laplacian matrix: A (B, N, N)
        '''
        if self_loop:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)

        deg_inv_sqrt = (torch.sum(A, 1) + 1e-5).pow(-0.5)

        LA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)

        return LA
