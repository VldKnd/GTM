import os
import sys
import math
import torch
import torch.nn as nn
from typing import Optional


class GTMEstimator(nn.Module):
    """
    Generative Topographic Mapping implementation with Gaussian Noise manifold model.
    Source: https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf
    """

    def __init__(self,  out_features: int, batch_size: int, in_features: Optional[int] = 2,
                 hidden_features: Optional[int] = 64, map_points: int = 10, verbose: bool = False,
                 lmbd: Optional[float] = None, cuda: bool=False):
        """
        Initialisation:
        Creates Class instance.

        :param in_features: Size of latent space
        :param out_features: Size of variable space
        :param map_points: Size of basis grid
        :param verbose: If true, the training process will output intermediate information
        """
        super(GTMEstimator, self).__init__()

        assert not (isinstance(in_features, type(None)) or isinstance(out_features, type(None))), \
            'You must provide sizes for hidden and variable dimension'
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.in_features = in_features
        self.out_features = out_features
        self.lmbd = lmbd
        self.W = nn.Sequential(nn.Linear(in_features, hidden_features),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_features, out_features, bias=False))
        self.bnorm = nn.BatchNorm1d(out_features)
        self.betta = torch.ones(1, requires_grad=True, device=self.device)

        self.grid = self.get_grid(map_points)
        self.verbose = verbose

        self.betta_opt = torch.optim.Adam([self.betta])
        self.w_opt = torch.optim.Adam(self.W.parameters(), lr=0.01)
        self.batch_size = batch_size

    def get_grid(self, n_points): # square grid only
        """
        Initializing basis grid for likelihood estimation

        :param n_points: size of the grid in the latent space
        :return: grid in the form of flattened torch tensor
        """
        grid = torch.meshgrid(*[torch.linspace(0, 1, n_points) for _ in range(self.in_features)])
        grid = torch.stack([t.flatten() for t in grid]).T.to(self.device)

        return grid

    def loss(self, p):
        p_x = torch.mean(p, dim=0)
        return (torch.clip(-1 * torch.log(p_x), min=torch.finfo(p.dtype).min)).sum(dim=0)

    def train_batch(self, batch):
        """
        Training function, to estimate the distributions sigma and mapping weights.

        :param X: Data from variable space
        :param batch_size: size of training batches
        :return: history of losses
        """

        self.train()
        self.W.zero_grad()
        self.betta_opt.zero_grad()
        self.w_opt.zero_grad()
        p = self.forward(batch)
        loss = self.loss(p)
        if self.lmbd:
            for params in self.y.parameters():
                if len(params.size()) == 2:
                    D1, D2 = params.size()
                    M = D1 * D2
                else:
                    D1, = params.size()
                    M = D1

                reg = torch.sum(torch.pow(params, 2))
                exp_reg = torch.exp((-self.lmbd / 2) * reg)
                p_w = exp_reg * (self.lmbd / (2 * math.pi)) ** (M / 2)
                loss += -1 * p_w

        loss.backward()
        self.betta_opt.step()
        self.w_opt.step()
        return loss.detach()

    def train_epoch(self, X):
        loss = []
        for i in range(math.ceil(X.shape[0]/self.batch_size)):
            batch = X[self.batch_size*i:self.batch_size*(i+1)]
            loss.append(self.train_batch(batch))
        return loss

    def transform(self, X, method: str = 'mean'):
        """
        Performs mapping from variable space to latent space.

        :param X: Data from variable space
        :param method: Defines projection type. Mean - using mean responsibility in grid
         gives mean position in latent dim,
         node - using max value of responsibility gives discrete position in latent dim.
        :return: Data in latent space
        """

        assert method in ('mean', 'node'), "Mode can be either mean or node."
        self.eval()
        with torch.no_grad():
            p = self.forward(X)
            p_x = p / p.sum(dim=0)
            if method == 'mean':
                return (self.grid.T @ p_x).T.detach()

            elif method == 'node':
                return self.grid[p_x.argmax(dim=0), :].detach()

    def inverse_transform(self, H):
        """
        Performs mapping from latent space to variable space.

        :param H: Data from latent space
        :return: Data in variable space
        """
        self.eval()
        return ((self.W(H) + self.bnorm.running_mean)*self.bnorm.running_var).detach()

    def forward(self, x):
        x = self.bnorm(x)
        dist = (-self.betta / 2) * torch.pow(torch.cdist(self.W(self.grid), x), 2)
        exp = torch.exp(dist)
        p = torch.pow(self.betta / (2 * math.pi), self.out_features / 2) * exp
        return p
