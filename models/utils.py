import torch
import numpy as np
import random
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def random_doc_batch(batch_size, min_num_sents, max_num_sents, min_sent_length,
                     max_sent_length, vocab_size, start_index=0):
    doc_batch = []
    max_doc_length = 0
    for i in range(batch_size):
        doc = []
        num_sents = random.randint(min_num_sents, max_num_sents)

        if num_sents > max_doc_length:
            max_doc_length = num_sents

        for s in range(num_sents):
            length = random.randint(min_sent_length, max_sent_length)
            doc.append([random.randint(start_index, vocab_size - 1) for i in range(length)])
        doc_batch.append(doc)

    doc_batch = sorted(doc_batch, key=len, reverse=True)
    doc_lengths = [len(d) for d in doc_batch]
    return doc_batch, doc_lengths, max_doc_length


#
def random_sentence_batch(batch_size, min_sent_length, max_sent_length, vocab_size, start_index=0,
                          stopword_label=False):
    sent_batch = []
    max_length = 0
    for i in range(batch_size):
        length = random.randint(min_sent_length, max_sent_length)
        sent_batch.append([random.randint(start_index, vocab_size - 1) for i in range(length)])
        if length > max_length:
            max_length = length
    # sent_batch = sorted(sent_batch, key=len, reverse=True)
    sent_lengths = [len(s) for s in sent_batch]
    if stopword_label:
        stopword_labels = []
        for s in sent_batch:
            label = [1 if random.random() > 0.8 else 0 for i in range(len(s))]
            stopword_labels.append(label)
        return sent_batch, sent_lengths, max_length, stopword_labels
    else:
        return sent_batch, sent_lengths, max_length


def random_identical_sentence_batch(batch_size, min_sent_length, max_sent_length, vocab_size, start_index=0):
    length = random.randint(min_sent_length, max_sent_length)
    sent_batch = [[random.randint(start_index, vocab_size - 1) for i in range(length)]] * batch_size
    sent_lengths = [length] * batch_size
    max_length = length
    return sent_batch, sent_lengths, max_length


#
def init_param(params):
    for p in params:
        p.data.uniform_(-0.1, 0.1)


#
def zero_init_rnn(batch_size, hidden_size, bidirectional=False,
                  impl_model='gru', gpu_device=None):
    if bidirectional:
        h = torch.zeros(2, batch_size, hidden_size)
        if impl_model == 'gru':
            if gpu_device is not None:
                h = h.to(gpu_device)
            return h, None
        else:  # lstm
            c = torch.zeros(2, batch_size, hidden_size)
            if gpu_device is not None:
                h = h.to(gpu_device)
                c = c.to(gpu_device)
            return h, c
    else:
        h = torch.zeros(1, batch_size, hidden_size)
        if impl_model == 'gru':
            if gpu_device is not None:
                h = h.to(gpu_device)
            return h, None
        else:  # lstm
            c = torch.zeros(1, batch_size, hidden_size)
            if gpu_device is not None:
                h = h.to(gpu_device)
                c = c.to(gpu_device)
            return h, c


#
def contains_nan(x):
    if torch.isnan(x).sum() > 0:
        return True
    return False


def check_nan(x, variable_name):
    if contains_nan(x):
        print(variable_name + ' contains nan')
        sys.exit()


def check_valid(x, variable_name):
    if torch.isnan(x).sum() > 0:
        print(variable_name, ' contains nan')
        sys.exit(1)

    num_elements = 1
    for d in x.shape:
        num_elements *= d
    if torch.isfinite(x).sum() < num_elements:
        print(variable_name, ' contains infinite')
        sys.exit(1)


def get_sentence_batch(doc_batch, sent_index):
    """

    :param doc_batch: list of doc, each doc is a list of sentences, each sentence is a list of word
                        docs are ordered from most #sentences to the least one
    :param sent_index:
    :return:
    """
    sent_batch = []
    for i in range(len(doc_batch)):
        if sent_index < len(doc_batch[i]):
            sent_batch.append(doc_batch[i][sent_index])
        else:
            break
    return sent_batch


def doc_transform(doc_batch):
    """
    transform docs from list of sentences to list of words format
    :param doc_batch:
    :return:
    """
    docs = []
    for d in doc_batch:
        words = []
        for s in d:
            words += s
        docs.append(words)
        # nw = len(words)
    return docs


def bow_batch_to_tensor(bow_batch, vocab_size, gpu_device=None):
    """
    convert a batch of docs' word vector into tensor
    :param bow_batch:
    :param vocab_size:
    :return:
    """

    batch_size = len(bow_batch)
    doc_array = np.array([np.bincount(bow_batch[d], minlength=vocab_size) for d in range(batch_size)])
    doc_tensor = torch.from_numpy(doc_array)
    if gpu_device is not None:
        doc_tensor = doc_tensor.to(gpu_device)
    return doc_tensor


def seq_batch_to_tensor(seq_batch, seq_lengths=None, gpu_device=None):
    batch_size = len(seq_batch)
    if seq_lengths is None:
        seq_lengths = [len(seq_batch[s]) for s in range(batch_size)]
    max_length = max(seq_lengths)
    seq_lengths = torch.LongTensor(seq_lengths)
    seq_tensor = torch.zeros(batch_size, max_length).long()
    if gpu_device is not None:
        seq_tensor = seq_tensor.to(gpu_device)
    for s in range(batch_size):
        seq_tensor[s, :seq_lengths[s]] = torch.LongTensor(seq_batch[s])
    return seq_tensor, seq_lengths, max_length


def create_mask(seq_lengths, max_length=None):
    batch_size = len(seq_lengths)
    if max_length is None:
        max_length = max(seq_lengths)
    mask = torch.zeros([batch_size, max_length])
    for i in range(batch_size):
        mask[i, :seq_lengths[i]] = 1
    return mask


def get_topwords(num_words, beta, vocab, stopword_indexes):
    beta[:, stopword_indexes] = -float('inf')
    _, indices = torch.topk(beta, num_words, dim=1)
    indices = indices.cpu().numpy()
    topwords = []
    num_topics = beta.shape[0]
    for z in range(num_topics):
        topwords.append([vocab[j] for j in list(indices[z])])
    return topwords


def scaledDP_attention(states, seq_lengths, W_q, W_k, W_v, gpu_device=None):
    """
    scaled dot-product attention mechanism described in "Attention is all you need", NIPS 2017
    :param states: tensor of shape (batch_size, max_sequence_length, embedding_size)
    :param seq_lengths: lengths of sequence in the batch
    :param W_q: query matrix of shape (embedding_size, output_size)
    :param W_k: key matrix of shape (embedding_size, output_size)
    :param W_v: value matrix of shape (embedding_size, output_size)
    :param gpu_device:
    :return: scaled dot-product attention of sequences in the b
    """
    batch_size = states.shape[0]
    output_size = W_v.shape[1]
    Q = torch.bmm(states, W_q.unsqueeze(0).expand(batch_size, -1, -1))
    K = torch.bmm(states, W_k.unsqueeze(0).expand(batch_size, -1, -1))
    V = torch.bmm(states, W_v.unsqueeze(0).expand(batch_size, -1, -1))
    dp = torch.mul(Q, K).sum(dim=2)
    # print('dp = ', dp.shape)
    att_weights = torch.softmax(dp, dim=1)
    # print('weights = ', att_weights.shape)
    mask = create_mask(seq_lengths)
    if gpu_device is not None:
        mask = mask.to(gpu_device)
    mask_weights = torch.mul(att_weights, mask)
    att = torch.mul(V, mask_weights.unsqueeze(2).expand(-1, -1, output_size)).sum(dim=1)
    return att


def yang_attention(states, seq_lengths, linear_W, linear_u, gpu_device=None):
    """
    the attention mechanism described in "Hierarchical Attention Networks for Document Classification",
    NAACL 2016
    :param states: tensor of shape (batch_size, max_sequence_length, embedding_size)
    :param seq_lengths: lengths of sequence in the batch
    :param linear_W: linear layer of size (embedding_size, context_size)
    :param linear_u: context vector of size (context_size)
    :param gpu_device:
    :return: attention of sequences in the batch
    """
    max_length, embedding_size = states.shape[1], states.shape[2]
    # batch of equation 5 in the paper
    contexts = linear_W(states)
    # check_nan(contexts, 'contexts')

    # print('contexts.shape = ', contexts.shape)
    # batch of elements in equation 6 in the paper
    logits = linear_u(contexts).squeeze(2)
    # print('logits.shape = ', logits.shape)
    # check_nan(logits, 'logits')

    # mask out elements at the end of shorter sentences
    mask = create_mask(seq_lengths, max_length)
    if gpu_device is not None:
        mask = mask.to(gpu_device)
    mask_weights = torch.mul(logits, mask)
    # check_nan(mask_weights, 'mask_weights')

    att_weights = mask_weights / mask_weights.sum(dim=1).unsqueeze(1).expand(-1, max_length)
    # check_nan(att_weights, 'att_weights')

    # batch of equation 7 in the paper
    att = torch.mul(states, att_weights.unsqueeze(2).expand(-1, -1, embedding_size)).sum(dim=1)
    return att


def normalize_topics(scores, stopword_indexes=None, gpu_device=None):
    num_topics, vocab_size = scores.shape
    mask = torch.ones(num_topics, vocab_size)
    if stopword_indexes is not None:
        mask[:, stopword_indexes] = 0
    if gpu_device is not None:
        mask = mask.to(gpu_device)
    exp = torch.exp(scores) * mask + 10e-12
    norm_sum = torch.sum(exp, dim=1)
    topics = exp / norm_sum.unsqueeze(1).expand(-1, vocab_size)
    return topics


def print_variable(var, var_name):
    print('\n', var_name, ' = ')
    print(var)
    print('\n')


def lm_loss(scores, doc_tensor, lengths, gpu_device=None):
    # print_variable(scores,'scores')
    exp = torch.exp(scores)
    # print_variable(exp, 'exp')
    mask = create_mask(lengths, doc_tensor.shape[1])
    if gpu_device:
        mask = mask.to(gpu_device)
    # print_variable(mask, 'mask')

    masked_exp = exp * (mask.unsqueeze(2).expand(-1, -1, scores.shape[2]))
    # print_variable(masked_exp, 'masked_exp')

    sum_exp = masked_exp.sum(dim=2) + 10E-12
    # print_variable(sum_exp,'sum_exp')
    s = exp / sum_exp.unsqueeze(2).expand(-1, -1, scores.shape[2])
    # print_variable(s, 's')
    log_s = torch.log(s).view(doc_tensor.shape[0] * doc_tensor.shape[1], -1)
    # print_variable(log_s, 'log_s')

    mask = mask.flatten().view(-1, 1)
    fd = doc_tensor.flatten().view(-1, 1)
    return (torch.gather(log_s, 1, fd) * mask).sum() / mask.sum()


def mullab_evaluation(features, labels, random_seed, is_sparse_matrix=False, proportion=None, cv=None, n_jobs=5,
                      binary_model=LogisticRegression(solver='liblinear')):
    """
    cross-validation for multi-label classification
    :param proportion:
    :param binary_model:
    :param n_jobs: #parallel threads
    :param features: numpy array
    :param labels: list of instances's label list
    :param cv:
    :return:
    """
    # print('in mullab')
    if proportion is None and cv is None:
        print('either proportion or cv should be not None')
        sys.exit()

    if random_seed is None:
        print('set the random_seed for consistent evaluation')
        sys.exit()

    labels = MultiLabelBinarizer().fit_transform(labels)
    num_instances = len(labels)
    indexes = [i for i in range(num_instances)]
    random.seed(random_seed)
    random.shuffle(indexes)

    # print('features.shape  = ', features.shape)
    # print('labels.shape = ', labels.shape)

    if proportion is not None:
        # print('proportion = ', proportion)
        train_size = int(proportion * len(indexes))
        train_data = indexes[:train_size]
        test_data = indexes[train_size:]

        train_labels = labels[train_data]
        test_labels = labels[test_data]

        train_data = features[train_data]
        test_data = features[test_data]

        #
        clf = OneVsRestClassifier(binary_model, n_jobs=n_jobs)
        if is_sparse_matrix:
            train_data.sort_indices()
        clf.fit(train_data, train_labels)
        #
        preds = clf.predict(test_data)
        #
        prec, rec, f1, support = None, None, None, None
        for l in range(labels.shape[1]):
            if sum(test_labels[:, l]) + sum(preds[:, l]) > 0:
                res = precision_recall_fscore_support(test_labels[:, l], preds[:, l], zero_division=0)
            else:
                res = (np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([len(test_labels[:, l]), 0]))
            if prec is None:
                prec = res[0].reshape(1, 2)
                rec = res[1].reshape(1, 2)
                f1 = res[2].reshape(1, 2)
                support = res[3].reshape(1, 2)
            else:
                prec = np.concatenate((prec, res[0].reshape(1, 2)), axis=0)
                rec = np.concatenate((rec, res[1].reshape(1, 2)), axis=0)
                f1 = np.concatenate((f1, res[2].reshape(1, 2)), axis=0)
                support = np.concatenate((support, res[3].reshape(1, 2)), axis=0)
        return prec, rec, f1, support
    else:
        # print('cv = ', cv)
        fold_size = num_instances // cv
        folds = []
        for f in range(cv - 1):
            folds.append(indexes[f * fold_size: (f + 1) * fold_size])
        folds.append(indexes[(cv - 1) * fold_size:])
        #
        prec, rec, f1, support = None, None, None, None
        for f in range(cv):
            # print('processing fold %d' % f)
            train_data = []
            for j in range(cv):
                if j != f:
                    train_data += folds[j]
            test_data = folds[f]

            train_labels = labels[train_data]
            test_labels = labels[test_data]

            train_data = features[train_data]
            test_data = features[test_data]

            #
            clf = OneVsRestClassifier(binary_model, n_jobs=n_jobs)
            clf.fit(train_data, train_labels)
            #
            preds = clf.predict(test_data)
            #
            for l in range(labels.shape[1]):
                if sum(test_labels[:, l]) + sum(preds[:, l]) > 0:
                    res = precision_recall_fscore_support(test_labels[:, l], preds[:, l], zero_division=0)
                else:
                    res = (np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array([len(test_labels[:, l]), 0]))
                if prec is None:
                    prec = res[0].reshape(1, 2)
                    rec = res[1].reshape(1, 2)
                    f1 = res[2].reshape(1, 2)
                    support = res[3].reshape(1, 2)
                else:
                    prec = np.concatenate((prec, res[0].reshape(1, 2)), axis=0)
                    rec = np.concatenate((rec, res[1].reshape(1, 2)), axis=0)
                    f1 = np.concatenate((f1, res[2].reshape(1, 2)), axis=0)
                    support = np.concatenate((support, res[3].reshape(1, 2)), axis=0)
        return prec, rec, f1, support


def coherence(words, dataset, normalized=True, wordIndex=True):
    print('word_set = ', words)
    word_set = set(words)
    counts = dict()
    for doc in dataset.documents:
        if wordIndex:
            unique_words = list(set(doc['bow']))
        else:
            unique_words = list(set([dataset.vocab[j] for j in doc['bow']]))
        # print('unique_words = ', unique_words)
        for i in range(len(unique_words)):
            w = unique_words[i]
            if w not in word_set:
                continue
            if w not in counts:
                counts[w] = 1
            else:
                counts[w] += 1
            for j in range(i + 1, len(unique_words), 1):
                v = unique_words[j]
                if v not in counts:
                    continue
                pair = (w, v)
                if v < w:
                    pair = (v, w)
                if pair not in counts:
                    counts[pair] = 1
                else:
                    counts[pair] += 1
    num_docs = len(dataset.documents)
    coh = 0
    for i in range(len(words)):
        pi = counts[words[i]] / num_docs
        for j in range(i + 1, (len(words)), 1):
            pj = counts[words[j]] / num_docs
            pair = (words[i], words[j])
            if words[j] < words[i]:
                pair = (words[j], words[i])
            if pair in counts:
                pij = counts[pair] / num_docs
                if normalized:
                    x = np.log(pij / (pi * pj))
                    y = np.log(pij)
                    # coh -= x / y
                    coh -= np.log(pij / (pi * pj)) / np.log(pij)
                    # print('%s %s pi = %f, pj = %f , pij = %f, x = %f, y = %f, x/y = %f, coh = %f' % (words[i], words[j],
                    #                                                                                 pi, pj, pij, x, y,
                    #                                                                                 x / y, coh))
                else:
                    coh += np.log(pij / (pi * pj))
    num_pairs = len(words) * (len(words) - 1) / 2
    print('coh = ', coh / num_pairs)
    return coh / num_pairs, counts


#
def reparameterize(mu, log_sigma):
    std = torch.exp(0.5 * log_sigma)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def kld_stdn(mus, log_variance):
    _, dim = mus.shape
    # kl_distance = -0.5 * torch.sum(1 + log_variance - torch.pow(mus, 2) - torch.exp(log_variance))
    kl_distance = 0.5 * (torch.sum(torch.exp(log_variance) + torch.pow(mus, 2) - log_variance, dim=1) - dim)
    # print('kl_loss = ', kl_distance)
    return kl_distance


def kld(mu1, log_variance1, mu2, log_variance2):
    """
    kl divergence between two gaussian distributions
    :param mu1:
    :param log_variance1:
    :param mu2:
    :param log_variance2:
    :return:
    """
    batch_size, dim = mu1.shape
    #
    a = torch.sum(log_variance2, dim=1) - torch.sum(log_variance1, dim=1)
    # print('a.shape = ', a.shape)
    #
    variance1 = torch.exp(log_variance1)
    variance2 = torch.exp(log_variance2)
    b = torch.sum(variance1 / variance2, dim=1)
    # print('b.shape = ', b.shape)
    #
    m = mu2 - mu1
    # print('m.shape = ', m.shape)
    c = torch.bmm((m / variance2).unsqueeze(2).transpose(1, 2), m.unsqueeze(2)).view(batch_size)
    # print('c.shape = ', c.shape)
    kl_divergence = 0.5 * (a - dim + b + c)
    return kl_divergence


def wasserstein_distance(mu1, log_variance1, mu2, log_variance2):
    """
    wasserstein distance between two gaussian distributions
    """
    distance = torch.sum((mu1 - mu2) ** 2, dim=1)
    variance1 = torch.exp(log_variance1)
    variance2 = torch.exp(log_variance2)
    distance += torch.sum(variance1, dim=1) + torch.sum(variance2, dim=1)
    distance -= 2 * torch.sum(torch.sqrt(variance1 * variance2), dim=1)
    return distance


def entropy(scores, is_logits=True, avg=True):
    if is_logits:
        scores = torch.softmax(scores, dim=1)

    ent = torch.distributions.Categorical(probs=scores).entropy()
    if avg:
        ent = ent.mean()
    return ent


def print_top_words(top_words, vocabulary):
    k = top_words.shape[1]
    for z in range(top_words.shape[0]):
        words = [vocabulary[top_words[z][i]] for i in range(k)]
        print('topic-%d: %s' % (z, ' '.join(words)))


def simplex_projection(x, z=1, gpu_device=None):
    """

    :param gpu_device:
    :param x: tensor of batch_size x dim size
    :param z: scalar
    :return:
    """

    bs, dim = x.shape
    mu = torch.sort(x, dim=1, descending=True)[0]
    cumsum = torch.cumsum(mu, dim=1)
    indices = torch.from_numpy(np.array([i + 1 for i in range(dim)])).unsqueeze(0).expand(bs, -1)
    if gpu_device is not None:
        indices = indices.to(gpu_device)
    y = mu * indices - (cumsum - z)
    rho = [(y[i, :] > 0).nonzero()[-1].item() for i in range(bs)]
    theta = torch.stack([cumsum[i, rho[i]].clone() for i in range(bs)]).view(-1, 1) - z
    rho = torch.from_numpy(np.array(rho) + 1).view(-1, 1)
    if gpu_device is not None:
        rho = rho.to(gpu_device)
    theta = theta / rho
    theta = theta.expand(-1, dim)
    if gpu_device is not None:
        v = torch.max(x - theta, torch.zeros(bs, dim).to(gpu_device))
    else:
        v = torch.max(x - theta, torch.zeros(bs, dim))
    return v


def get_propotion(x, topk):
    top_elements = x.argsort()[:, -topk:]
    # print(top_elements)
    top_elements = np.array([x[z, top_elements[z, :]] for z in range(x.shape[0])])
    # print(top_elements)
    # print(top_elements.sum(axis=1))
    # print(x.sum(axis=1))
    props = top_elements.sum(axis=1) / x.sum(axis=1)
    return props


def pairwise_cosine_similarity(x):
    dot = torch.matmul(x, x.transpose(0, 1))
    norm = torch.norm(x, 2, 1) + 1E-10
    sim = dot / norm.unsqueeze(0)
    sim = sim / norm.unsqueeze(1)
    return sim


def topic_diversity(topic_vectors):
    num_topics = topic_vectors.shape[0]
    sim = pairwise_cosine_similarity(topic_vectors)
    # if torch.isnan(sim.sum()):
    #     print('sim is nan')
    #     sys.exit(1)

    # inspect(sim, 'z')
    # check_valid(sim, 'z')

    z = sim[torch.triu(torch.ones(num_topics, num_topics), diagonal=1) == 1]
    # if z.max() > 1:
    #     print('z > 1')

    # inspect(z, 'z')
    # check_valid(z, 'z')

    z = torch.acos(z)
    # if torch.isnan(z.sum()):
    #     print('z is nan')
    #     sys.exit(1)

    # inspect(z, 'acos-z')
    # check_valid(z, 'acos-z')

    zeta = z.mean()

    # inspect(zeta, 'zeta')
    # check_valid(zeta, 'zeta')

    nu = z.var()

    # inspect(nu, 'nu')
    # check_valid(nu, 'nu')
    # print(zeta.item(), nu.item())

    return nu - zeta


def inspect(x, variable_name=None):
    if len(x.shape) == 1:
        print('inspecting: {}: shape = {} max = {} min = {} max-min = {}'.format(variable_name,
                                                                                 x.shape,
                                                                                 x.max(),
                                                                                 x.min(),
                                                                                 x.max() - x.min()))
    elif len(x.shape) == 2:
        print('inspecting: {}: shape = {} max = {} min = {} max(max-min) = {}'.format(variable_name,
                                                                                      x.shape,
                                                                                      x.max(),
                                                                                      x.min(),
                                                                                      (x.max(dim=1)[0] - x.min(dim=1)[
                                                                                          0]).max()))
    else:
        pass
