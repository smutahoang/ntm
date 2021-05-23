from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
import numpy as np


def online_nmf(dataset, num_topics, num_iterations, batchsize=512):
    """

    :param dataset:
    :param num_topics:
    :param num_iterations:
    :return: model, vocab, documents' topic distribution, topics' word distribution, and doc-2-bow transformer
    """
    corpus = [[dataset.vocab[w] for w in d['bow']] for d in dataset.documents]
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    model = Nmf(corpus, num_topics=num_topics, passes=num_iterations, id2word=dictionary, chunksize=batchsize)
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
