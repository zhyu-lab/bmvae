import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import math


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def kl_loss(z_mean, z_stddev):
    return torch.mean(-0.5 * torch.sum(1 + 2*torch.log(z_stddev) - z_mean ** 2 - z_stddev ** 2, dim=1), dim=0)


def ce_loss(predict, target, tv):
    loss = torch.nn.BCELoss()
    return loss(predict[tv], target[tv])


def xs_gen(cell_list, batch_size, random):
    if random == 1:
         np.random.shuffle(cell_list)
    steps = math.ceil(len(cell_list) / batch_size)
    for i in range(steps):
        batch_x = cell_list[i * batch_size: i * batch_size + batch_size]
        yield i, batch_x


class Encoder(torch.nn.Module):
    def __init__(self, d_in, h1, h2, d_out):
        super(Encoder, self).__init__()
        self.enc_layer1 = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(),
        )
        self.enc_layer2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(),
        )

        self.enc_mu = nn.Linear(h2, d_out)
        self.enc_log_sigma = nn.Linear(h2, d_out)

        initialize_weights(self)


    def forward(self, x):
        x = self.enc_layer1(x)
        return self.enc_layer2(x)


class Decoder(torch.nn.Module):
    def __init__(self, d_in, h1, h2, d_out):
        super(Decoder, self).__init__()
        self.dec_layer1 = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(),
        )
        self.dec_layer2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(),
        )
        self.dec_layer3 = nn.Linear(h2, d_out)
        initialize_weights(self)


    def forward(self, x):
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        return torch.sigmoid(self.dec_layer3(x))


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, dimension):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.encoder.enc_mu(h_enc)
        mu = mu.cuda()
        log_sigma = self.encoder.enc_log_sigma(h_enc)
        log_sigma = log_sigma.cuda()
        sigma = torch.exp(log_sigma)


        self.z_mean = mu
        self.z_sigma = sigma

        eps = torch.randn_like(sigma)
        return mu + eps * sigma


    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)
