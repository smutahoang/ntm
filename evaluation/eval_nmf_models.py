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


def topic_coherence(topic_word_matrix, dataset, dictionary, top_k):
    coherence = 0.0
    for t in topic_word_matrix:
        words = t.argsort()[-top_k:]
        words = [dictionary[w] for w in words]
        coherence += utils.coherence(words, dataset, wordIndex=False)[0]
    return coherence / topic_word_matrix.shape[0]


def get_docs_classification_performance(document_topic_matrix, dataset, random_seed, num_folds):
    labels = [dataset.documents[i]['labels'] for i in range(len(dataset.documents))]
    return utils.mullab_evaluation(document_topic_matrix, labels, random_seed=random_seed, cv=num_folds)
