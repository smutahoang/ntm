import sys
import os
import numpy as np

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


def avg_unseen_docs_perplexity(model_name, model, dataset, dictionary):
    if model_name == 'online_lda':
        corpus = [[dataset.vocab[w] for w in d['bow']] for d in dataset.documents]
        corpus = [dictionary.doc2bow(text) for text in corpus]
        avg_perplexity = model.bound(corpus) / len(corpus)
        return avg_perplexity
    else:
        print('unseen docs perplexity is not defined for model {}'.format(model_name))


def avg_unseen_words_perplexity(model_name, topic_distributions, topic_word_distributions, dataset, dictionary):
    if model_name == 'online_lda':
        corpus = [[dictionary.token2id[dataset.vocab[w]] for w in d['test_bow']] for d in dataset.documents]
        vec = np.matmul(topic_distributions, topic_word_distributions)
        vec = np.log(vec + 1E-12)
        avg_perplexity = 0.0
        for d in range(len(corpus)):
            avg_perplexity += vec[d][corpus[d]].sum()
        # avg_perplexity = avg_perplexity / sum([len(d) for d in corpus])
        avg_perplexity = avg_perplexity / len(corpus)
        return -avg_perplexity
    elif model_name == 'gibbs_lda':
        corpus = [[dictionary.token2id[dataset.vocab[w]] for w in d['test_bow']] for d in dataset.documents]
        vec = np.matmul(topic_distributions, topic_word_distributions)
        vec = np.log(vec + 1E-12)
        avg_perplexity = 0.0
        for d in range(len(corpus)):
            avg_perplexity += vec[d][corpus[d]].sum()
        # avg_perplexity = avg_perplexity / sum([len(d) for d in corpus])
        avg_perplexity = avg_perplexity / len(corpus)
        return -avg_perplexity
    else:
        print('unseen words perplexity is not defined for model {}'.format(model_name))


def topic_coherence(topic_word_distributions, dataset, vocab, top_k):
    coherence = 0.0
    for t in topic_word_distributions:
        words = t.argsort()[-top_k:]
        words = [vocab[w] for w in words]
        coherence += utils.coherence(words, dataset, wordIndex=False)[0]
    return coherence / topic_word_distributions.shape[0]


def get_docs_classification_performance(topic_distributions, dataset, num_folds, random_seed):
    labels = [dataset.documents[i]['labels'] for i in range(len(dataset.documents))]
    return utils.mullab_evaluation(topic_distributions, labels, random_seed=random_seed, cv=num_folds)
