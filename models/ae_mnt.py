# autoencoder mnt: All-In-One file

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


class AEMNT(nn.Module):

    def __init__(self, net_arch):
        super(AEMNT, self).__init__()
        self.net_arch = net_arch
        # encoder
        self.mlp = nn.Sequential(nn.Linear(net_arch['num_input'], net_arch['n_hidden']),
                                 nn.ReLU(),
                                 nn.Linear(net_arch['n_hidden'], net_arch['n_hidden']),
                                 nn.ReLU())
        self.en_drop = nn.Dropout(0.8)

        self.mean_layer = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])
        self.mean_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for mean

        self.theta_drop = nn.Dropout(0.2)

        # topic decoder
        self.decoder = nn.Linear(net_arch['num_topic'], net_arch['num_input'])
        self.decoder_bn = nn.BatchNorm1d(net_arch['num_input'])

        # initialize decoder weight
        if net_arch['init_mult'] != 0:
            # xavier init
            std = 1. / math.sqrt(net_arch['init_mult'] * (net_arch['num_input'] + net_arch['num_topic']))
            self.decoder.weight.data.normal_(0, std)
            self.decoder.bias.data.normal_(0, std)

        self.topic_diversity_reg = net_arch['topic_diversity_reg']
        self.topic_sparsity_reg = net_arch['topic_sparsity_reg']
        self.doc_sparsity_reg = net_arch['doc_sparsity_reg']

    def forward(self, doc_freq_vecs, is_train=True):
        # utils.inspect(self.decoder.weight, 'decoder_weight')
        # utils.check_valid(self.decoder.weight, 'decoder_weight')
        #
        # utils.inspect(self.word_vectors, 'word_vectors')
        # utils.check_valid(self.word_vectors, 'word_vectors')

        enc = self.mlp(doc_freq_vecs)
        if is_train:
            enc = self.en_drop(enc)

        # utils.inspect(enc, 'enc')
        # utils.check_valid(enc, 'enc')

        mu = self.mean_layer(enc)
        mu = self.mean_bn(mu)
        if is_train:
            mu = self.theta_drop(mu)
        theta = torch.softmax(mu, dim=1)

        # if is_train:
        #    theta = self.theta_drop(theta)
        # if is_train:
        #    theta = self.theta_drop(theta)
        # utils.inspect(theta, 'theta')
        # utils.check_valid(theta, 'theta')

        logits = self.decoder_bn(self.decoder(theta))
        # logits = self.decoder(theta)

        recon = torch.softmax(logits, dim=1)
        # utils.inspect(recon, 'recon')
        # utils.check_valid(recon, 'recon')

        trace = {'doc_freq_vecs': doc_freq_vecs,
                 'theta': theta,
                 # 'topic_logits': topic_logits,
                 # 'topic_logits_bn': topic_logits_bn,
                 # 'beta': beta,
                 'recon': recon}

        if is_train:
            return self.loss(doc_freq_vecs, recon, theta, is_train=True)  # , trace
        else:
            return enc, theta, recon, self.loss(doc_freq_vecs, recon, theta, is_train=False)

    def loss(self, doc_freq_vecs, recon, theta, is_train=True):
        # negative loglikelihood
        nl = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
        # utils.inspect(nl, 'nl')
        # utils.check_valid(nl, 'nl')

        if is_train:
            nl = nl.mean()
            num_topics = theta.shape[1]
            num_words = recon.shape[1]
            # self.decoder_bn.eval()
            # beta = torch.softmax(self.decoder_bn(self.decoder(torch.eye(num_topics).to(self.net_arch['device']))), 1)
            beta = torch.softmax(self.decoder(torch.eye(num_topics).to(self.net_arch['device'])), 1)
            # self.decoder_bn.train()
            # topic diversity regularization

            topic_diversity = utils.topic_diversity(torch.relu(beta - 1 / num_words))
            # utils.inspect(topic_diversity, 'topic_diversity')
            # utils.check_valid(topic_diversity, 'topic_diversity')

            topic_sparsity = utils.entropy(beta, is_logits=False, avg=False)
            # utils.inspect(topic_sparsity, 'topic_sparsity')
            # utils.check_valid(topic_sparsity, 'topic_sparsity')

            doc_sparsity, _ = torch.max(theta, 1)
            doc_sparsity = -doc_sparsity

            reg = self.topic_diversity_reg * topic_diversity + self.topic_sparsity_reg * topic_sparsity
            reg = reg.mean() + (self.doc_sparsity_reg * doc_sparsity).mean()
            # utils.inspect(reg, 'reg')
            # utils.check_valid(reg, 'reg')

            return nl + reg, nl, reg

        else:
            return nl
