import sys
import os
import pickle

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

from models import lda_models


def train(dataset, dataset_name, model_name, num_topics, num_iterations,
          save=False, output_path=None, save_filename=None):
    if save:
        if output_path is None or save_filename is None:
            print('output_path and save_filename are not specified')
        sys.exit()

    if model_name == 'online_lda':
        model, vocab, theta, beta, dictionary = lda_models.online_lda(dataset, num_topics=num_topics,
                                                                      num_iterations=num_iterations)
        run = {'dataset': dataset_name,
               'model_name': model_name,
               'model': model,
               'num_topics': num_topics,
               'theta': theta,
               'beta': beta,
               'dictionary': dictionary,
               'vocab': vocab}
        if save:
            pickle.dump(run, open('{}/{}'.format(output_path, save_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return run
    elif model_name == 'gibbs_lda':
        model, vocab, theta, beta, word2index = lda_models.gibbs_lda(dataset, num_topics=num_topics,
                                                                     num_iterations=num_iterations)
        labels = [dataset.documents[i]['labels'] for i in range(len(dataset.documents))]
        run = {'dataset': dataset_name,
               'model_name': model_name,
               'model': model,
               'num_topics': num_topics,
               'theta': theta,
               'beta': beta,
               'word2index': word2index,
               'labels': labels,
               'vocab': vocab}
        if save:
            pickle.dump(run, open('{}/{}'.format(output_path, save_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return run
    elif model_name == 'biterm_lda':
        model, vocab, theta, beta = lda_models.biterm_lda(dataset, num_topics=num_topics, num_iterations=num_iterations)
        run = {'dataset': dataset_name,
               'model_name': model_name,
               'model': model,
               'num_topics': num_topics,
               'theta': theta,
               'beta': beta,
               'vocab': vocab}
        if save:
            pickle.dump(run, open('{}/{}'.format(output_path, save_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return run
    else:
        print('model {} is not yet supported'.format(model_name))
