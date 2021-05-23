from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import lda  # https://pypi.org/project/lda/
from biterm.btm import oBTM  # https://pypi.org/project/biterm/
from itertools import combinations
from scipy.sparse import csr_matrix
import numpy as np


def online_lda(dataset, num_topics, num_iterations):
    """

    :param dataset:
    :param num_topics:
    :param num_iterations:
    :return: model, vocab, documents' topic distribution, topics' word distribution, and doc-2-bow transformer
    """
    corpus = [[dataset.vocab[w] for w in d['bow']] for d in dataset.documents]
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    model = LdaModel(corpus, num_topics=num_topics, iterations=num_iterations, id2word=dictionary)
    thetas = model.get_document_topics(corpus, minimum_probability=0.0)
    # convert to numpy array
    topic_distributions = []
    for i in range(len(thetas)):
        theta = thetas[i]
        distribution = [0.0] * num_topics
        for e in theta:
            distribution[e[0]] = e[1]
        topic_distributions.append(distribution)
    topic_distributions = np.array(topic_distributions)
    return model, [dictionary[i] for i in range(len(dictionary))], topic_distributions, model.get_topics(), dictionary


def gibbs_lda(dataset, num_topics, num_iterations):
    """

    :param dataset:
    :param num_topics:
    :param num_iterations:
    :return: model, vocab, documents' topic distribution, and topics' word distribution, and word-2-id dictionary
    """
    corpus = [[dataset.vocab[w] for w in d['bow']] for d in dataset.documents]
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    rows = []
    cols = []
    counts = []
    for d in range(len(corpus)):
        row = [d] * len(corpus[d])
        col = [e[0] for e in corpus[d]]
        count = [e[1] for e in corpus[d]]

        rows.extend(row)
        cols.extend(col)
        counts.extend(count)
    corpus = csr_matrix((counts, (rows, cols)))
    model = lda.LDA(n_topics=num_topics, n_iter=num_iterations)
    topic_distributions = model.fit_transform(corpus)
    return model, [dictionary[i] for i in range(len(dictionary))], topic_distributions, model.topic_word_, dictionary


def biterm_lda(dataset, num_topics, num_iterations):
    """
    :return: model, vocab, documents' topic distribution, and topics' word distribution
    :param dataset:
    :param num_topics:
    :param num_iterations:
    :return:
    """
    biterms = [[b for b in combinations(dataset.documents[d]['bow'], 2)] for d in range(len(dataset.documents))]
    btm = oBTM(num_topics=num_topics, V=dataset.vocab)
    topic_distributions = btm.fit_transform(biterms, iterations=num_iterations)
    return btm, dataset.vocab, topic_distributions, btm.phi_wz.T
