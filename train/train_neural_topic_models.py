import numpy as np
import torch
import torch.cuda
import torch.autograd
import math
import sys
import os
import copy

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

from models.prodLDA import ProdLDA
from models.nvdm import NVDM
from models.gsm import GSM
from models.nsmtm import NSMTM
from models.nvctm import NVCTM
from models.scholar import torchScholar
from models.gaton import GATON
from models.ae_mnt import AEMNT


def default_params(model, num_epoch=100, batch_size=512, gpu_index=0):
    params = {'model': model,
              'batch_size': batch_size,
              'optimizer': 'Adam',
              'learning_rate': 0.002,
              'momentum': 0.99,
              'num_epoch': num_epoch,
              'init_mult': 0.001,
              'device': torch.device('cuda:{}'.format(gpu_index))}
    if model == 'NVDM':  # https://arxiv.org/pdf/1511.06038.pdf
        params['n_hidden'] = 500

    elif model == 'GSM':  # https://arxiv.org/pdf/1706.00359.pdf
        params['embedding_size'] = 128
        params['n_hidden'] = 500
        params['vi_hidden'] = 256
        params['reg_lambda'] = 0.1

    elif model == 'ProdLDA' or model == 'NVLDA':  # https://arxiv.org/abs/1703.01488
        params['n_hidden'] = 500
        params[
            'dir_prior'] = 0.02  # following https://github.com/akashgit/autoencoding_vi_for_topic_models/blob/master/run.py#L41

    elif model == 'NSMDM' or model == 'NSMTM':  # https://arxiv.org/pdf/1810.09079.pdf
        params['n_hidden'] = 500
        params['embedding_size'] = 128
        params['learning_rate'] = 5e-5

    elif model == 'NVCTM':  # https://dl.acm.org/doi/10.1145/3308558.3313561
        params['n_hidden'] = 500
        params['ctf_length'] = 4

    elif model == 'Scholar':  # https://arxiv.org/abs/1705.09296
        params['n_labels'] = 0
        params['n_prior_covars'] = 0
        params['n_topic_covars'] = 0
        params['classifier_layers'] = 0
        params['use_interactions'] = False
        params['l1_beta_reg'] = 1.0
        params['l1_beta_c_reg'] = 1.0
        params['l1_beta_ci_reg'] = 1.0
        params['l2_prior_reg'] = 0.0
        params['embedding_dim'] = 300
        pass  # TODO
    elif model == 'AEMNT':
        params['en1_units'] = 500
        params['en2_units'] = 500
        params['embedding_dim'] = 300
        params['topic_diversity_reg'] = 1.0
        params['topic_sparsity_reg'] = 0.05
        params['doc_sparsity_reg'] = 0.5
    else:
        pass

    return params


def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)


def make_model(params):
    if params['model'] == 'ProdLDA' or params['model'] == 'NVLDA':
        model = ProdLDA(params)
    elif params['model'] == 'NVDM':
        model = NVDM(params)
    elif params['model'] == 'GSM':
        model = GSM(params)
    elif params['model'] == 'NSMDM' or params['model'] == 'NSMTM':
        model = NSMTM(params)
    elif params['model'] == 'NVCTM':
        model = NVCTM(params)
    elif params['model'] == 'Scholar':
        alpha = 1.0 * np.ones((1, params['num_topic'])).astype(np.float32)
        model = torchScholar(params, alpha, device=params['device'])
    elif params['model'] == 'AEMNT':
        model = AEMNT(params)
    else:
        print('model {} is not yet supported'.format(params['model']))
        sys.exit(1)

    model.to(params['device'])
    return model


def make_optimizer(model, params):
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'], betas=(params['momentum'], 0.999))
    elif params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), params['learning_rate'], momentum=params['momentum'])
    else:
        assert False, 'Unknown optimizer {}'.format(params['optimizer'])
    return optimizer


def train(dataset, dataset_name, num_topics, params, save=False, output_path=None):
    vocab_size = len(dataset.vocab)
    num_docs = len(dataset.documents)
    params['num_input'] = vocab_size
    params['num_topic'] = num_topics

    model = make_model(params)
    optimizer = make_optimizer(model, params)
    best_loss = math.inf
    best_states = None
    best_epoch = -1
    for epoch in range(params['num_epoch']):
        all_indices = torch.randperm(len(dataset.documents)).split(params['batch_size'])
        loss_epoch = 0.0
        rec_epoch = 0.0
        kld_epoch = 0.0
        model.train()  # switch to training mode
        num_batch = 0
        for batch_indices in all_indices:
            batch_indices = batch_indices.to(params['device'])
            # input = Variable(tensor_tr[batch_indices])
            doc_batch = dataset.get_data_batch(batch_indices)
            bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
            bows = [to_onehot(bows[i], vocab_size) for i in range(len(bows))]
            doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(params['device'])
            # with torch.autograd.detect_anomaly():
            if params['model'] == 'Scholar':
                if params['l1_beta_reg'] > 0:
                    l1_beta = 0.5 * np.ones([vocab_size, num_topics], dtype=np.float32) / float(num_docs)
                else:
                    l1_beta = None
                loss, rec, kld = model(doc_freq_vecs, None, None, None, l1_beta=l1_beta)
            else:
                loss, rec, kld = model(doc_freq_vecs)
                # loss, trace = model(doc_freq_vecs)
            # loss, rec, kld = loss
            # print('e = %d, b = %d, kld = %f, rec = %f, loss = %f' % (
            #     epoch, num_batch, kld.item(), rec.item(), loss.item()))
            # print(loss)
            # optimize
            optimizer.zero_grad()  # clear previous gradients
            # if torch.isnan(loss):
            #     print('--train: loss is nan')
            #     sys.exit(1)
            # if not torch.isfinite(loss):
            #     print('--train: loss is infinite')
            #     sys.exit(1)

            loss.backward()  # backprop

            # for p in model.parameters():
            #     if torch.isnan(p.grad.sum()) > 0 or torch.isnan(p.sum()) > 0:
            #         return model, trace

            optimizer.step()  # update parameters
            # report
            loss_epoch += loss.item()  # add loss to loss_epoch
            rec_epoch += rec.item()
            kld_epoch += kld.item()
            num_batch += 1
        if loss_epoch / len(all_indices) < best_loss:
            best_states = copy.deepcopy(model.state_dict())
            for s in best_states:
                best_states[s] = best_states[s].cpu()
            best_loss = loss_epoch / len(all_indices)
            best_epoch = epoch
        if epoch % 1 == 0:
            print('Epoch {}, loss={}, rec = {}, kld = {}'.format(epoch, loss_epoch / len(all_indices),
                                                                 rec_epoch / len(all_indices),
                                                                 kld_epoch / len(all_indices)))
        if save:
            # save model
            save_path = '%s/%s_%d_%d' % (output_path, params['model'], num_topics, epoch)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_path)

    model.cpu()
    model.load_state_dict(best_states)
    model.to(params['device'])
    run = {'dataset': dataset_name,
           'model_name': params['model'],
           'num_topics': num_topics,
           'model': model,
           'best_loss': best_loss,
           'best_epoch': best_epoch
           }
    return run
