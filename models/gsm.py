# GSM from https://arxiv.org/pdf/1706.00359.pdf

import torch
import torch.nn as nn
import sys
import os
import math

# find path to root directory of the project so as to import from other packages
# to be refactored
# print('current script: storage/toy_datasets/imdb.py')
# print('os.path.abspath(__file__) = ', os.path.abspath(__file__))
tokens = os.path.abspath(__file__).split('/')
# print('tokens = ', tokens)
path2root = '/'.join(tokens[:-3])
# print('path2root = ', path2root)
if path2root not in sys.path:
    sys.path.append(path2root)

from models import utils


class GSM(nn.Module):

    def __init__(self, net_arch):
        super(GSM, self).__init__()
        self.net_arch = net_arch
        # encoder
        self.mlp = nn.Sequential(nn.Linear(net_arch['num_input'], net_arch['n_hidden']),
                                 nn.Tanh(),
                                 nn.Linear(net_arch['n_hidden'], net_arch['n_hidden']),
                                 nn.Tanh())
        self.mlp_drop = nn.Dropout(0.8)

        self.mean_fc = nn.Linear(net_arch['n_hidden'], net_arch['vi_hidden'])
        self.logvar_fc = nn.Linear(net_arch['n_hidden'], net_arch['vi_hidden'])
        self.gsm = nn.Linear(net_arch['vi_hidden'], net_arch['num_topic'])

        topic_vectors = nn.Parameter(torch.rand(net_arch['num_topic'], net_arch['embedding_size']))
        word_vectors = nn.Parameter(torch.rand(net_arch['embedding_size'], net_arch['num_input']))

        self.register_parameter('topic_vectors', topic_vectors)
        self.register_parameter('word_vectors', word_vectors)

        # prior mean and variance as constant buffers

        prior_mean = torch.Tensor(1, net_arch['vi_hidden']).fill_(0)
        prior_var = torch.Tensor(1, net_arch['vi_hidden']).fill_(1)
        prior_logvar = torch.log(prior_var)

        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_logvar', prior_logvar)

        # initialize decoder weight
        if net_arch['init_mult'] != 0:
            # self.topic_vectors.data.uniform_(0, net_arch['init_mult'])
            # self.word_vectors.data.uniform_(0, net_arch['init_mult'])

            # xavier init
            std = 1. / math.sqrt(net_arch['init_mult'] * (net_arch['num_input'] + net_arch['num_topic']))
            self.topic_vectors.data.normal_(0, std)
            self.word_vectors.data.normal_(0, std)

            self.reg_lambda = net_arch['reg_lambda']

    def forward(self, doc_freq_vecs, avg_loss=True, is_train=True):
        # compute posterior
        en_vec = self.mlp(doc_freq_vecs)
        if is_train:
            en_vec = self.mlp_drop(en_vec)

        posterior_mean = self.mean_fc(en_vec)

        posterior_logvar = self.logvar_fc(en_vec)  # posterior log variance
        posterior_var = posterior_logvar.exp()

        # take sample
        if is_train:
            eps = doc_freq_vecs.data.new().resize_as_(posterior_mean.data).normal_()  # noise
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization

            p = torch.softmax(self.gsm(z), 1)
            recon = torch.matmul(p, torch.softmax(torch.matmul(self.topic_vectors, self.word_vectors), 1))

            return self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            pre_softmax = self.gsm(posterior_mean)
            p = torch.softmax(pre_softmax, 1)
            recon = torch.matmul(p, torch.softmax(torch.matmul(self.topic_vectors, self.word_vectors), 1))
            return posterior_mean, pre_softmax, p, recon, self.loss(doc_freq_vecs, recon, posterior_mean,
                                                                    posterior_logvar,
                                                                    posterior_var, avg_loss, topic_regularization=False)

    def loss(self, doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, avg=True,
             topic_regularization=True):
        # NL
        NL = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
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
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.net_arch['vi_hidden'])
        # print('KLD = ', KLD)
        # loss
        loss = (NL + KLD)

        if topic_regularization:
            loss += self.reg_lambda * utils.topic_diversity(self.topic_vectors)
        # loss = test_var.sum(dim=1)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD
