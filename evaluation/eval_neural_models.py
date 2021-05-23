import sys
import os
import numpy as np
import torch
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

from models import utils
from train import train_neural_topic_models as neural_trainer


def avg_unseen_docs_perplexity(model_name, model, dataset, batch_size=512):
    vocab_size = len(dataset.vocab)
    model.train(False)
    model.eval()
    with torch.no_grad():
        if model_name == 'NVDM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'GSM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NVLDA':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'ProdLDA':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'Scholar':
            # all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            all_indices = torch.tensor([i for i in range(len(dataset.documents))]).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.device)
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.device)
                _, _, _, _, loss = model(doc_freq_vecs, None, None, None, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NSMDM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NSMTM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NVCTM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, loss = model(doc_freq_vecs, avg_loss=False, is_train=False)
                loss = loss[0].cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'AEMNT':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, loss = model(doc_freq_vecs, is_train=False)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        else:
            print('perplexity is not defined for model {}'.format(model_name))


def avg_unseen_words_perplexity(model_name, model, dataset, batch_size=512):
    vocab_size = len(dataset.vocab)
    model.train(False)
    model.eval()
    with torch.no_grad():
        if model_name == 'NVDM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'GSM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NVLDA':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'ProdLDA':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'Scholar':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.device)
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.device)
                _, _, recon, _, loss = model(doc_freq_vecs, None, None, None, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.device)

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NSMDM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NSMTM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'NVCTM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, recon, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        elif model_name == 'AEMNT':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            perplexity = None
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                _, _, recon, _ = model(doc_freq_vecs, is_train=False)

                bows = [doc_batch[d]['test_bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])

                loss = -(doc_freq_vecs * (recon + 1e-10).log()).sum(1)
                loss = loss.cpu().detach().numpy()
                if perplexity is None:
                    perplexity = loss
                else:
                    perplexity = np.concatenate((perplexity, loss), axis=0)
            return perplexity.mean()
        else:
            print('perplexity is not defined for model {}'.format(model_name))


def topic_coherence(model_name, model, dataset, top_k):
    model.train(False)
    model.eval()
    if model_name == 'NVDM':
        beta = copy.deepcopy(model.decoder.weight.transpose(0, 1).cpu().detach().numpy())
        beta[:, list(dataset.stopword_indexes)] = -10E12
        coherence = 0.0
        for t in beta:
            words = t.argsort()[-top_k:]
            words = [dataset.vocab[w] for w in words]
            coherence += utils.coherence(words, dataset, wordIndex=False)[0]
        return coherence / beta.shape[0]
    elif model_name == 'GSM':
        with torch.no_grad():
            beta = torch.matmul(model.topic_vectors, model.word_vectors).cpu().detach().numpy()
            beta[:, list(dataset.stopword_indexes)] = -10E12
            coherence = 0.0
            for t in beta:
                words = t.argsort()[-top_k:]
                words = [dataset.vocab[w] for w in words]
                coherence += utils.coherence(words, dataset, wordIndex=False)[0]
            return coherence / beta.shape[0]
    elif model_name == 'NVLDA':
        with torch.no_grad():
            beta = model.decoder_bn(model.decoder.weight.transpose(0, 1))
            beta = beta.cpu().detach().numpy()
            beta[:, list(dataset.stopword_indexes)] = -10E12
            coherence = 0.0
            for t in beta:
                words = t.argsort()[-top_k:]
                words = [dataset.vocab[w] for w in words]
                coherence += utils.coherence(words, dataset, wordIndex=False)[0]
            return coherence / beta.shape[0]
    elif model_name == 'ProdLDA':
        with torch.no_grad():
            e = torch.eye(model.net_arch['num_topic']).to(model.net_arch['device'])
            z = torch.zeros(model.net_arch['num_topic'], model.net_arch['num_topic']).to(model.net_arch['device'])
            beta = model.decoder_bn(model.decoder(e)) - model.decoder_bn(model.decoder(z))
            beta = beta.cpu().detach().numpy()
            beta[:, list(dataset.stopword_indexes)] = -10E12
            coherence = 0.0
            for t in beta:
                words = t.argsort()[-top_k:]
                words = [dataset.vocab[w] for w in words]
                coherence += utils.coherence(words, dataset, wordIndex=False)[0]
            return coherence / beta.shape[0]
    elif model_name == 'Scholar':
        e = torch.eye(model.n_topics).to(model.device)
        z = torch.zeros(model.n_topics, model.n_topics).to(model.device)
        beta = model.eta_bn_layer(model.beta_layer(e)) - model.eta_bn_layer(model.beta_layer(z))
        # beta = model.beta_layer.weight.to('cpu').detach().numpy().T  # follows original code
        # print('beta.shape = ', beta.shape)
        beta[:, list(dataset.stopword_indexes)] = -10E12
        coherence = 0.0
        for t in beta:
            words = t.argsort()[-top_k:]
            words = [dataset.vocab[w] for w in words]
            coherence += utils.coherence(words, dataset, wordIndex=False)[0]
        return coherence / beta.shape[0]
    elif model_name == 'NSMDM':
        beta = torch.matmul(model.topic_vectors, model.word_vectors).cpu().detach().numpy()
        beta[:, list(dataset.stopword_indexes)] = -10E12
        coherence = 0.0
        for t in beta:
            words = t.argsort()[-top_k:]
            words = [dataset.vocab[w] for w in words]
            coherence += utils.coherence(words, dataset, wordIndex=False)[0]
        return coherence / beta.shape[0]
    elif model_name == 'NSMTM':
        beta = model.beta_sparsemax(torch.matmul(model.topic_vectors, model.word_vectors)).cpu().detach().numpy()
        beta[:, list(dataset.stopword_indexes)] = -10E12
        coherence = 0.0
        for t in beta:
            words = t.argsort()[-top_k:]
            words = [dataset.vocab[w] for w in words]
            coherence += utils.coherence(words, dataset, wordIndex=False)[0]
        return coherence / beta.shape[0]
    elif model_name == 'NVCTM':
        beta = copy.deepcopy(model.decoder.weight.transpose(0, 1).cpu().detach().numpy())
        beta[:, list(dataset.stopword_indexes)] = -10E12
        coherence = 0.0
        for t in beta:
            words = t.argsort()[-top_k:]
            words = [dataset.vocab[w] for w in words]
            coherence += utils.coherence(words, dataset, wordIndex=False)[0]
        return coherence / beta.shape[0]
    elif model_name == 'AEMNT':
        beta = torch.softmax(model.decoder(torch.eye(model.net_arch['num_topic']).to(model.net_arch['device'])), 1)
        beta[:, list(dataset.stopword_indexes)] = -10E12
        coherence = 0.0
        for t in beta:
            words = t.argsort()[-top_k:]
            words = [dataset.vocab[w] for w in words]
            coherence += utils.coherence(words, dataset, wordIndex=False)[0]
        return coherence / beta.shape[0]
    elif model_name == 'GATON':
        pass
    else:
        print('perplexity is not defined for model {}'.format(model_name))


def get_docs_classification_performance(model_name, model, dataset, num_folds, random_seed, batch_size=512):
    vocab_size = len(dataset.vocab)
    model.train(False)
    model.eval()
    with torch.no_grad():
        if model_name == 'NVDM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    unnormalized_topic_features = posterior_mean
                else:
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, posterior_mean), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            run = {'num_options': 1,
                   'performance': utils.mullab_evaluation(unnormalized_topic_features, labels,
                                                          random_seed=random_seed, cv=num_folds)}
            return run
        elif model_name == 'GSM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)

            enc_features = None
            pre_topic_features = None
            unnormalized_topic_features = None

            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, pre_softmax, p, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)

                posterior_mean = posterior_mean.cpu().detach().numpy()
                pre_softmax = pre_softmax.cpu().detach().numpy()
                p = p.cpu().detach().numpy()

                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    pre_topic_features = pre_softmax
                    unnormalized_topic_features = p
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    pre_topic_features = np.concatenate((pre_topic_features, pre_softmax), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, p), axis=0)

                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            pre_topic_performance = utils.mullab_evaluation(pre_topic_features, labels, random_seed=random_seed,
                                                            cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)

            run = {'num_options': 3,
                   'performance': [enc_performance, pre_topic_performance, topic_performance]}
            return run
        elif model_name == 'NVLDA':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, p, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                p = p.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    unnormalized_topic_features = p
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, p), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'ProdLDA':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, p, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                p = p.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    unnormalized_topic_features = p
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, p), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'Scholar':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.device)
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.device)
                posterior_mean, p, _, _, _ = model(doc_freq_vecs, None, None, None, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                p = p.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    unnormalized_topic_features = p
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, p), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'NSMDM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, p, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                p = p.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    unnormalized_topic_features = p
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, p), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'NSMTM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, p, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                p = p.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    unnormalized_topic_features = p
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, p), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'NVCTM':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            unnormalized_topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                posterior_mean, z, _, _ = model(doc_freq_vecs, avg_loss=False, is_train=False)
                posterior_mean = posterior_mean.cpu().detach().numpy()
                z = z.cpu().detach().numpy()
                if unnormalized_topic_features is None:
                    enc_features = posterior_mean
                    unnormalized_topic_features = z
                else:
                    enc_features = np.concatenate((enc_features, posterior_mean), axis=0)
                    unnormalized_topic_features = np.concatenate((unnormalized_topic_features, z), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(unnormalized_topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'AEMNT':
            all_indices = torch.randperm(len(dataset.documents)).split(batch_size)
            enc_features = None
            topic_features = None
            labels = []
            for batch_indices in all_indices:
                batch_indices = batch_indices.to(model.net_arch['device'])
                doc_batch = dataset.get_data_batch(batch_indices)
                bows = [doc_batch[d]['bow'] for d in range(len(doc_batch))]
                bows = [neural_trainer.to_onehot(bows[i], vocab_size) for i in range(len(bows))]
                doc_freq_vecs = torch.from_numpy(np.array(bows)).float().to(model.net_arch['device'])
                enc, z, _, _ = model(doc_freq_vecs, is_train=False)
                enc = enc.cpu().detach().numpy()
                z = z.cpu().detach().numpy()
                if topic_features is None:
                    enc_features = enc
                    topic_features = z
                else:
                    enc_features = np.concatenate((enc_features, enc), axis=0)
                    topic_features = np.concatenate((topic_features, z), axis=0)
                labels.extend([dataset.documents[i]['labels'] for i in batch_indices])
            enc_performance = utils.mullab_evaluation(enc_features, labels, random_seed=random_seed, cv=num_folds)
            topic_performance = utils.mullab_evaluation(topic_features, labels, random_seed=random_seed,
                                                        cv=num_folds)
            run = {'num_options': 2,
                   'performance': [enc_performance, topic_performance]}
            return run
        elif model_name == 'GATON':
            pass
        else:
            print('perplexity is not defined for model {}'.format(model_name))
