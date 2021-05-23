# NSMDM and NSMTM from
# Sparsemax and Relaxed Wasserstein for Topic Sparsity, WSDM 2019

import sys
import os

import torch
import torch.nn as nn
import math

# find path to root directory of the project so as to import from other packages
# to be refactored
# print('current script: storage/builtin_datasets.py')
# print('os.path.abspath(__file__) = ', os.path.abspath(__file__))
tokens = os.path.abspath(__file__).split('/')
# print('tokens = ', tokens)
path2root = '/'.join(tokens[:-2])
# print('path2root = ', path2root)
if path2root not in sys.path:
    sys.path.append(path2root)

from models.sparsemax import Sparsemax
from models import utils


class NSMTM(nn.Module):

    def __init__(self, net_arch):
        super(NSMTM, self).__init__()
        self.net_arch = net_arch
        # encoder
        self.mlp = nn.Sequential(nn.Linear(net_arch['num_input'], net_arch['n_hidden']),
                                 nn.Tanh(),
                                 nn.Linear(net_arch['n_hidden'], net_arch['n_hidden']),
                                 nn.Tanh())

        self.mlp_dropout = nn.Dropout(0.8)

        self.mean_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])
        self.mean_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for mean
        self.logvar_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])
        self.logvar_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for logvar

        #
        self.theta_sparsemax = Sparsemax(dim=1, gpu_device=net_arch['device'])
        if self.net_arch['model'] == 'NSMTM':
            self.beta_sparsemax = Sparsemax(dim=1, gpu_device=net_arch['device'])

        # decoder
        topic_vectors = nn.Parameter(torch.rand(net_arch['num_topic'], net_arch['embedding_size']))
        word_vectors = nn.Parameter(torch.rand(net_arch['embedding_size'], net_arch['num_input']))

        self.register_parameter('topic_vectors', topic_vectors)
        self.register_parameter('word_vectors', word_vectors)

        prior_mean = torch.Tensor(1, net_arch['num_topic']).fill_(0)
        prior_var = torch.Tensor(1, net_arch['num_topic']).fill_(1)
        prior_logvar = torch.log(prior_var)

        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)

        # initialize decoder weight
        if net_arch['init_mult'] != 0:
            if self.net_arch['model'] == 'NSMDM':
                # xavier init
                std = 1. / math.sqrt(net_arch['init_mult'] * (net_arch['num_input'] + net_arch['num_topic']))
                self.topic_vectors.data.normal_(0, std)
                self.word_vectors.data.normal_(0, std)

            else:
                self.topic_vectors.data.uniform_(0, net_arch['init_mult'])
                self.word_vectors.data.uniform_(0, net_arch['init_mult'])

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
            p = self.theta_sparsemax(z)

            # do reconstruction
            if self.net_arch['model'] == 'NSMDM':
                recon = torch.softmax(torch.matmul(p, torch.matmul(self.topic_vectors, self.word_vectors)), 1)
            elif self.net_arch['model'] == 'NSMTM':
                recon = torch.matmul(p, self.beta_sparsemax(torch.matmul(self.topic_vectors, self.word_vectors)))
                # print(recon.sum(dim=1))
            else:
                print('model {} is not yet supported'.format(self.net_arch['model']))
            return self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar, avg_loss)
        else:
            p = self.theta_sparsemax(posterior_mean)
            # do reconstruction
            if self.net_arch['model'] == 'NSMDM':
                recon = torch.softmax(torch.matmul(p, torch.matmul(self.topic_vectors, self.word_vectors)), 1)
            else:  # NSMTM
                recon = torch.matmul(p, self.beta_sparsemax(torch.matmul(self.topic_vectors, self.word_vectors)))

            return posterior_mean, p, recon, self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar, avg_loss)

    def loss(self, doc_freq_vecs, recon, posterior_mean, posterior_logvar, avg=True):
        # NL
        NL = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
        # WD
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)

        wd = utils.wasserstein_distance(posterior_mean, posterior_logvar, prior_mean, prior_logvar)
        # loss
        loss = (NL + wd)
        # loss = test_var.sum(dim=1)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean(), NL.mean(), wd.mean()
        else:
            return loss, NL, wd
