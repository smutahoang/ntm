# Neural Variational Correlated Topic Model
# from Neural Variational Correlated Topic Model
# https://dl.acm.org/doi/10.1145/3308558.3313561

import torch
import torch.nn as nn
import math


def compute_household_matrix(v, beye):
    """

    :param v:
    :param beye:
    :return:
    """
    # print('\t\t v.shape = ', v.shape)
    # print('\t\t beye.shape = ', beye.shape)
    norm_squared = torch.sum(v * v, dim=1)
    x = v.unsqueeze(2)
    x = torch.bmm(x, x.transpose(1, 2))
    h = beye - 2 * x / norm_squared.unsqueeze(1).unsqueeze(1)
    return h


class NVCTM(nn.Module):

    def __init__(self, net_arch):
        super(NVCTM, self).__init__()
        self.net_arch = net_arch
        # encoder
        self.mlp = nn.Sequential(nn.Linear(net_arch['num_input'], net_arch['n_hidden']),
                                 nn.Tanh(),
                                 nn.Linear(net_arch['n_hidden'], net_arch['n_hidden']),
                                 nn.Tanh())

        self.mlp_dropout = nn.Dropout(0.8)

        self.mean_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])  # 100  -> 50
        self.mean_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for mean
        self.logvar_fc = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])  # 100  -> 50
        self.logvar_bn = nn.BatchNorm1d(net_arch['num_topic'])  # bn for logvar

        self.v0 = nn.Linear(net_arch['n_hidden'], net_arch['num_topic'])
        self.ctf_length = net_arch['ctf_length']

        # decoder
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

        # remove BN's scale parameters
        # self.logvar_bn.weight.requires_grad = False
        # self.logvar_bn.bias.requires_grad = False

        # self.mean_bn.weight.requires_grad = False
        # self.mean_bn.bias.requires_grad = False

        # self.decoder_bn.weight.requires_grad = False
        # self.decoder_bn.bias.requires_grad = False

    def forward(self, doc_freq_vecs, avg_loss=True, is_train=True):
        # print('doc_freq_vecs.shape = ', doc_freq_vecs.shape)
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

            # create batch of identity matrix
            batch_size, dim_size = posterior_mean.shape
            # print('batch_size, dim_size = ', batch_size, dim_size)

            beye = torch.eye(dim_size)
            beye = beye.reshape(1, dim_size, dim_size)
            beye = beye.repeat(batch_size, 1, 1)
            beye = beye.to(self.net_arch['device'])

            # print('beye.shape = ', beye.shape)

            household_matrix = compute_household_matrix(self.v0(en_vec), beye)  # initial household matrix
            # print('household_matrix.shape = ', household_matrix.shape)

            transformer_u = household_matrix
            for i in range(self.ctf_length):
                if i == 0:
                    v = posterior_mean
                else:
                    v = torch.bmm(household_matrix, v.unsqueeze(2)).squeeze(2)
                # print('\ti = ', i)
                # print('\tv.shape = ', v.shape)
                # print('\thousehold_matrix.shape = ', household_matrix.shape)

                household_matrix = compute_household_matrix(v, beye)
                transformer_u = torch.bmm(transformer_u, household_matrix)
            z = torch.bmm(transformer_u, z.unsqueeze(2)).squeeze(2)
            # do reconstruction
            recon = torch.softmax(self.decoder(z), 1)
            return self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, transformer_u,
                             avg_loss)
        else:
            # create batch of identity matrix
            batch_size, dim_size = posterior_mean.shape
            # print('batch_size, dim_size = ', batch_size, dim_size)
            beye = torch.eye(dim_size)
            beye = beye.reshape(1, dim_size, dim_size)
            beye = beye.repeat(batch_size, 1, 1)
            beye = beye.to(self.net_arch['device'])

            # print('beye.shape = ', beye.shape)

            household_matrix = compute_household_matrix(self.v0(en_vec), beye)  # initial household matrix
            # print('household_matrix.shape = ', household_matrix.shape)

            transformer_u = household_matrix
            for i in range(self.ctf_length):
                if i == 0:
                    v = posterior_mean
                else:
                    v = torch.bmm(household_matrix, v.unsqueeze(2)).squeeze(2)
                # print('\ti = ', i)
                # print('\tv.shape = ', v.shape)
                # print('\thousehold_matrix.shape = ', household_matrix.shape)

                household_matrix = compute_household_matrix(v, beye)
                transformer_u = torch.bmm(transformer_u, household_matrix)
            z = torch.bmm(transformer_u, posterior_mean.unsqueeze(2)).squeeze(2)
            recon = torch.softmax(self.decoder(z), 1)
            return posterior_mean, z, recon, self.loss(doc_freq_vecs, recon, posterior_mean, posterior_logvar,
                                                       posterior_var, transformer_u,
                                                       avg_loss)

    def loss(self, doc_freq_vecs, recon, posterior_mean, posterior_logvar, posterior_var, transformer_u, avg=True):
        # NL
        NL = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)

        # print('LOSS\t NL.shape = ', NL.shape)

        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_mean)

        prior_logvar = self.prior_logvar.expand_as(posterior_mean)

        # print('LOSS\t transformer_u.shape = ', transformer_u.shape)
        # print('LOSS\t posterior_mean.shape = ', posterior_mean.shape)

        trace_term = torch.bmm(transformer_u, torch.diag_embed(posterior_var))
        trace_term = torch.bmm(trace_term, transformer_u)

        # print('LOSS\t var_division.shape = ', trace_term.shape)
        # print('LOSS\t prior_var.shape = ', prior_var.shape)

        trace_term = torch.bmm(trace_term, torch.diag_embed(1 / prior_var))

        trace_term = torch.diagonal(trace_term, dim1=1, dim2=2).sum(dim=1)
        # print('LOSS\t trace_term.shape = ', trace_term.shape)

        # var_division = torch.bmm(torch.bmm(transformer_u, posterior_var), transformer_u) / prior_var

        diff = torch.bmm(transformer_u, posterior_mean.unsqueeze(2)).squeeze(2) - prior_mean

        diff_term = torch.sum(diff * diff / prior_var, dim=1)
        # print('LOSS\t diff_term.shape = ', diff_term.shape)

        logvar_division = torch.sum(prior_logvar - posterior_logvar, dim=1)
        # print('LOSS\t logvar_division.shape = ', logvar_division.shape)

        # put KLD together
        KLD = 0.5 * (trace_term + diff_term + logvar_division - self.net_arch['num_topic'])

        # print('KLD = ', KLD)
        # loss
        loss = (NL + KLD)
        # loss = test_var.sum(dim=1)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD
