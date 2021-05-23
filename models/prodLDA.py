# ProdLDA and NVLDA models
# from Akash Srivastava and Charles Sutton, 2017,
# https://arxiv.org/pdf/1703.01488.pdf

import torch
import torch.nn as nn
import math


class ProdLDA(nn.Module):

    def __init__(self, net_arch):
        super(ProdLDA, self).__init__()
        self.net_arch = net_arch
        # encoder
        self.mlp = nn.Sequential(nn.Linear(net_arch['num_input'], net_arch['n_hidden']),
                                 nn.Softplus(),
                                 nn.Linear(net_arch['n_hidden'], net_arch['n_hidden']),
                                 nn.Softplus())

        self.mlp_dropout = nn.Dropout(0.8)

        self.mean_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])  # 100  -> 50
        self.mean_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for mean
        self.logvar_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])  # 100  -> 50
        self.logvar_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for logvar
        # z
        self.p_drop = nn.Dropout(0.2)
        # decoder
        self.decoder = nn.Linear(net_arch['num_topic'], net_arch['num_input'])  # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(net_arch['num_input'])  # bn for decoder
        # prior mean and variance as constant buffers
        prior_mean = torch.Tensor(1, net_arch['num_topic']).fill_(0)
        # prior_var = torch.Tensor(1, net_arch['num_topic']).fill_(net_arch['variance'])
        dir_prior = net_arch['dir_prior']  # Dirichlet prior
        num_topic = net_arch['num_topic']
        prior_var = torch.Tensor(1, net_arch['num_topic']).fill_((num_topic - 1) / (num_topic * dir_prior))
        prior_logvar = torch.log(prior_var)

        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)

        # initialize decoder weight
        if net_arch['init_mult'] != 0:
            # self.decoder.weight.data.uniform_(0, net_arch['init_mult'])

            # xavier init
            std = 1. / math.sqrt(net_arch['init_mult'] * (net_arch['num_input'] + net_arch['num_topic']))
            self.decoder.weight.data.normal_(0, std)

    def forward(self, doc_freq_vecs, avg_loss=True, is_train=True):
        # compute posterior
        en_vec = self.mlp(doc_freq_vecs)
        if is_train:
            en_vec = self.mlp_dropout(en_vec)

        posterior_mean = self.mean_bn(self.mean_fc(en_vec))  # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en_vec))  # posterior log variance
        posterior_var = posterior_logvar.exp()

        if is_train:
            # take sample
            # eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_())  # noise
            eps = doc_freq_vecs.data.new().resize_as_(posterior_mean.data).normal_()  # noise
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            p = torch.softmax(z, dim=1)  # mixture probability
            p = self.p_drop(p)
            # do reconstruction
            if self.net_arch['model'] == 'prodLDA':
                recon = torch.softmax(self.decoder_bn(self.decoder(p)), dim=1)
            else:  # NVLDA
                recon = torch.matmul(p, torch.softmax(self.decoder_bn(self.decoder.weight.transpose(0, 1)), 1))
            return self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            p = torch.softmax(posterior_mean, dim=1)  # mixture probability
            # do reconstruction
            if self.net_arch['model'] == 'prodLDA':
                recon = torch.softmax(self.decoder_bn(self.decoder(p)), dim=1)
            else:  # NVLDA
                recon = torch.matmul(p, torch.softmax(self.decoder_bn(self.decoder.weight.transpose(0, 1)), 1))
            return posterior_mean, p, recon, self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar,
                                                       posterior_var,
                                                       avg_loss)

    def loss(self, doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
        # print('LOSS\t NL.shape = ', NL.shape)
        # print('NL = ', NL)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        # prior_mean = Variable(self.prior_mean).expand_as(posterior_mean)
        # prior_var = Variable(self.prior_var).expand_as(posterior_mean)
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_mean)
        # print('prior_mean = ', prior_mean)
        # prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        # print('prior_logvar = ', prior_logvar)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together

        # print('LOSS\t var_division.shape = ', var_division.shape)

        # print('LOSS\t diff_term.shape = ', diff_term.shape)
        # print('LOSS\t logvar_division.shape = ', logvar_division.shape)

        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.net_arch['num_topic'])
        # print('KLD = ', KLD)
        # loss
        loss = (NL + KLD)
        # loss = test_var.sum(dim=1)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD
