import os
import sys
import math
import torch
import torch.nn as nn


class GTMEstimator(nn.Module):
    """
    Generative Topographic Mapping implementation with Gaussian Noise manifold model.
    Source: https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf
    """

    def __init__(self, in_features=None, out_features=None, n_x_points=100, verbose=False, y=None, method="mean",
                 lmbd=0, cuda=False):
        """
        Initialisation:
        Creates Class instance.

        :param in_features: Size of latent space
        :param out_features: Size of variable space
        :param n_x_points: Size of basis grid
        :param verbose: If true, the training process will output intermediate information
        :param y: Mapping function, has to be differentiable by pytorch.
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

        if isinstance(y, type(None)):
            self.y = nn.Linear(in_features, out_features)

        self.y = y

        self.betta = torch.rand(1, requires_grad=True, device=self.device)

        self.grid = self.get_grid(n_x_points)
        self.verbose = verbose

        self.betta_opt = torch.optim.Adam([self.betta])
        self.y_opt = torch.optim.Adam(self.y.parameters())

        self.method = method

        self.mean = 0
        self.std = 1

    def get_grid(self, n_points):
        """
        Initializing basis grid for likelihood estimation

        :param n_points: size of the grid in the latent space
        :return: grid in the form of flattened torch tensor
        """
        grid = torch.meshgrid(*[torch.linspace(0, 1, n_points) for _ in range(self.in_features)])
        grid = torch.stack([t.flatten() for t in grid]).T.to(self.device)

        return grid

    def train_epoch(self, X, batch_size=256):
        """
        Training function, to estimate the distributions sigma and mapping weights.

        :param X: Data from variable space
        :param batch_size: size of training batches
        :return: history of losses
        """

        l_h = []  # Loss history
        n_x_variable, D = X.size()

        X = (X - self.mean)/self.std

        for i in range(math.ceil(n_x_variable / batch_size)):
            self.y.zero_grad()
            self.betta_opt.zero_grad()
            self.y_opt.zero_grad()

            batch = X[i * batch_size:(i + 1) * batch_size]
            size, _ = batch.size()

            dist = (-self.betta / 2) * torch.pow(torch.cdist(self.y(self.grid), batch), 2)
            exp = torch.exp(dist)
            p = torch.pow(self.betta / (2 * math.pi), D / 2) * exp
            p_x = torch.mean(p, dim=0)
            loss = (torch.clip(-1 * torch.log(p_x), min=torch.finfo(p.dtype).min)).sum(dim=0)

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
            self.y_opt.step()

            l_h.append(loss.item() / size)

        return l_h

    def train_epochs(self, X, n_epochs=1, batch_size=256):
        """
        Training function, to estimate the distributions sigma and mapping weights.
        Runs one epoch procedure `n_epochs` times.

        :param X: Data from variable space
        :param n_epochs: number of epochs to run
        :param batch_size: size of training batches
        :return: history of losses
        """

        l_h = []

        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)

        for i in range(n_epochs):
            l_h.extend(self.train_epoch(X, batch_size))

            if self.verbose:
                print('epoch #{}: likelihood: {:.3f} betta: {:.3f}'.format(i + 1, l_h[-1], self.betta.item()))

        return l_h

    def transform(self, X):
        """
        Performs mapping from variable space to latent space.

        :param X: Data from variable space
        :return: Data in latent space
        """

        assert self.method in ('mean', 'mode'), "Mode can be either mean or mode."

        with torch.no_grad():
            n_x_variable, D = X.size()

            dist = (-self.betta / 2) * torch.pow(torch.cdist(self.y(self.grid), (X - self.mean)/self.std), 2)
            exp = torch.exp(dist)
            p = torch.pow(self.betta / (2 * math.pi), D / 2) * exp
            p_x = p / p.sum(dim=0)

            if self.method == 'mean':
                return (self.grid.T @ p_x).T

            elif self.method == 'mode':
                return self.grid[p_x.argmax(dim=0), :]

    def inverse_transform(self, H):
        """
        Performs mapping from latent space to variable space.

        :param H: Data from latent space
        :return: Data in variable space
        """

        return (self.y(H) + self.mean)*self.std


if __name__ == "__main__":
    in_features = 2
    hidden_features = 8
    out_features = 4

    y = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features))

    est = GTMEstimator(in_features=in_features, out_features=out_features, n_x_points=3, verbose=True, y=y)

    Df = torch.rand(1024, 4)
    est.train_epochs(Df, 1000)
    output = (est.transform(Df)).numpy()
