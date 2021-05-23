# NVDM from https://arxiv.org/pdf/1511.06038.pdf

import torch
import torch.nn as nn
import math


class NVDM(nn.Module):

    def __init__(self, net_arch):
        super(NVDM, self).__init__()
        self.net_arch = net_arch
        # encoder
        self.mlp = nn.Sequential(nn.Linear(net_arch['num_input'], net_arch['n_hidden']),
                                 nn.Tanh(),
                                 nn.Linear(net_arch['n_hidden'], net_arch['n_hidden']),
                                 nn.Tanh())
        self.mlp_drop = nn.Dropout(0.8)
        self.mean_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])

        self.logvar_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])
        nn.init.zeros_(self.logvar_fc.weight)  # following https://github.com/ysmiao/nvdm/blob/master/nvdm.py#L51
        nn.init.zeros_(self.logvar_fc.bias)

        self.decoder = nn.Linear(net_arch['num_topic'], net_arch['num_input'])  # 50   -> 1995

        # prior mean and variance as constant buffers

        prior_mean = torch.Tensor(1, net_arch['num_topic']).fill_(0)
        prior_var = torch.Tensor(1, net_arch['num_topic']).fill_(1)
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
            en_vec = self.mlp_drop(en_vec)

        posterior_mean = self.mean_fc(en_vec)

        posterior_logvar = self.logvar_fc(en_vec)  # posterior log variance
        posterior_var = posterior_logvar.exp()

        # take sample
        if is_train:
            eps = doc_freq_vecs.data.new().resize_as_(posterior_mean.data).normal_()  # noise
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            recon = torch.softmax(self.decoder(z), dim=1)
            return self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)

        else:
            recon = torch.softmax(self.decoder(posterior_mean), dim=1)
            return posterior_mean, recon, self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar,
                                                    posterior_var, avg_loss)

    def loss(self, doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
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
