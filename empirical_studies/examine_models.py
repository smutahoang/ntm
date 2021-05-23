import sys
import os
import pickle
from joblib import Parallel, delayed

# find path to root directory of the project so as to import from other packages
# to be refactored
# print('current script: storage/toy_datasets/imdb.py')
# print('os.path.abspath(__file__) = ', os.path.abspath(__file__))
tokens = os.path.abspath(__file__).split('/')
# print('tokens = ', tokens)
path2root = '/'.join(tokens[:-2])
# print('path2root = ', path2root)
if path2root not in sys.path:
    sys.path.append(path2root)

from data.data_loader import Dataset
from train import train_lda_models as lda_trainer
from train import train_nmf_models as nmf_trainer
from train import train_neural_topic_models as neural_trainer
from evaluation import eval_lda_models as lda_evaluator
from evaluation import eval_nmf_models as nmf_evaluator
from evaluation import eval_neural_models as neural_evaluator
import random
import copy

num_runs = 10
num_iterations = 200
top_ks = [5, 10]  # top_k for topic coherence measurement
num_folds = 5  # for cross-evaluation in document classification task


def unseen_doc_perplexity(model, dataset, num_topics, gpu_index=0):
    if model not in {'online_lda', 'NVDM', 'GSM', 'NVLDA', 'ProdLDA', 'Scholar', 'NSMDM', 'NSMTM', 'NVCTM'}:
        print('unseen docs perplexity is not defined for model {}'.format(model))
        return None
    avg_perplexity = []
    for i in range(num_runs):
        print('run ', i)
        random.seed(i)  # for consistent train-test split among runs
        indexes = [i for i in range(len(dataset.documents))]
        random.shuffle(indexes)
        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)
        num_train_docs = int(0.9 * len(train_dataset.documents))
        train_dataset.documents = [train_dataset.documents[i] for i in indexes[:num_train_docs]]
        test_dataset.documents = [test_dataset.documents[i] for i in indexes[num_train_docs:]]

        if model == 'online_lda':
            run = lda_trainer.train(dataset=train_dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            avg_perplexity.append(lda_evaluator.avg_unseen_docs_perplexity(model, run['model'], test_dataset,
                                                                           run['dictionary']))
        elif model in {'NVDM', 'GSM', 'NVLDA', 'ProdLDA', 'Scholar', 'NSMDM', 'NSMTM', 'NVCTM'}:
            params = neural_trainer.default_params(model=model, num_epoch=num_iterations, gpu_index=gpu_index)
            run = neural_trainer.train(dataset=train_dataset, dataset_name='None', num_topics=num_topics, params=params)
            avg_perplexity.append(neural_evaluator.avg_unseen_docs_perplexity(model, run['model'], test_dataset))

        else:
            print('perplexity is not defined for model {}'.format(model))
    return avg_perplexity


def unseen_words_perplexity(model, dataset, num_topics, gpu_index=0):
    if model not in {'online_lda', 'gibbs_lda',
                     'NVDM', 'GSM', 'NVLDA', 'ProdLDA', 'Scholar', 'NSMDM', 'NSMTM', 'NVCTM'}:
        print('unseen words perplexity is not defined for model {}'.format(model))
        return None
    avg_perplexity = []
    for i in range(num_runs):
        print('run ', i)
        fold_dataset = copy.deepcopy(dataset)
        fold_dataset.split_bow(random_seed=i)  # for consistent train-test split among runs

        if model == 'online_lda':
            run = lda_trainer.train(dataset=fold_dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            avg_perplexity.append(lda_evaluator.avg_unseen_words_perplexity(model, run['theta'], run['beta'],
                                                                            fold_dataset, run['dictionary']))

        elif model == 'gibbs_lda':
            run = lda_trainer.train(dataset=fold_dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            avg_perplexity.append(lda_evaluator.avg_unseen_words_perplexity(model, run['theta'], run['beta'],
                                                                            fold_dataset, run['word2index']))
        elif model in {'NVDM', 'GSM', 'NVLDA', 'ProdLDA', 'Scholar', 'NSMDM', 'NSMTM', 'NVCTM'}:
            params = neural_trainer.default_params(model=model, num_epoch=num_iterations, gpu_index=gpu_index)
            run = neural_trainer.train(dataset=fold_dataset, dataset_name='None', num_topics=num_topics, params=params)
            avg_perplexity.append(neural_evaluator.avg_unseen_words_perplexity(model, run['model'], fold_dataset))
        else:
            print('perplexity is not defined for model {}'.format(model))
    return avg_perplexity


def topic_coherence(model, dataset, num_topics, gpu_index=0):
    coherences = []
    for i in range(num_runs):
        print('run ', i)
        random.seed(i)  # new random start

        if model == 'online_lda':
            run = lda_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            coherence = []
            for k in top_ks:
                c = lda_evaluator.topic_coherence(run['beta'], dataset, run['dictionary'], k)
                coherence.append(c)
            coherences.append(coherence)

        elif model == 'gibbs_lda':
            run = lda_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            coherence = []
            for k in top_ks:
                c = lda_evaluator.topic_coherence(topic_word_distributions=run['beta'],
                                                  dataset=dataset, vocab=run['word2index'], top_k=k)
                coherence.append(c)
            coherences.append(coherence)
        elif model == 'biterm_lda':
            run = lda_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            coherence = []
            for k in top_ks:
                c = lda_evaluator.topic_coherence(topic_word_distributions=run['beta'],
                                                  dataset=dataset, vocab=run['vocab'], top_k=k)
                coherence.append(c)
            coherences.append(coherence)

        elif model == 'nmf':
            run = nmf_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            coherence = []
            for k in top_ks:
                c = nmf_evaluator.topic_coherence(topic_word_matrix=run['beta'], dataset=dataset,
                                                  dictionary=run['vocab'],
                                                  top_k=k)
                coherence.append(c)
            coherences.append(coherence)

        elif model in {'NVDM', 'GSM', 'NVLDA', 'ProdLDA', 'Scholar', 'NSMDM', 'NSMTM', 'NVCTM'}:
            params = neural_trainer.default_params(model=model, num_epoch=num_iterations, gpu_index=gpu_index)
            run = neural_trainer.train(dataset=dataset, dataset_name='None', num_topics=num_topics, params=params)
            coherence = []
            for k in top_ks:
                c = neural_evaluator.topic_coherence(model_name=model, model=run['model'], dataset=dataset, top_k=k)
                coherence.append(c)
            coherences.append(coherence)
        else:
            print('topic_coherence is not defined for model {}'.format(model))
    return coherences


def doc_classification(model, dataset, num_topics, gpu_index=0):
    prec, rec, f1 = None, None, None
    for i in range(num_runs):
        print('dc:run ', i)
        random.seed(i)  # new random start
        if model == 'online_lda':
            run = lda_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            p, r, f, _ = lda_evaluator.get_docs_classification_performance(run['theta'], dataset, num_folds=num_folds,
                                                                           random_seed=i)
            if prec is None:
                prec, rec, f1 = [], [], []

            prec.append(p[:, 1].mean())
            rec.append(r[:, 1].mean())
            f1.append(f[:, 1].mean())

        elif model == 'gibbs_lda':
            run = lda_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            p, r, f, _ = lda_evaluator.get_docs_classification_performance(run['theta'], dataset, num_folds=num_folds,
                                                                           random_seed=i)
            if prec is None:
                prec, rec, f1 = [], [], []
            prec.append(p[:, 1].mean())
            rec.append(r[:, 1].mean())
            f1.append(f[:, 1].mean())

        elif model == 'biterm_lda':
            run = lda_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)
            p, r, f, _ = lda_evaluator.get_docs_classification_performance(run['theta'], dataset,
                                                                           num_folds=num_folds, random_seed=i)
            if prec is None:
                prec, rec, f1 = [], [], []
            prec.append(p[:, 1].mean())
            rec.append(r[:, 1].mean())
            f1.append(f[:, 1].mean())

        elif model == 'nmf':
            run = nmf_trainer.train(dataset=dataset, dataset_name='None',
                                    model_name=model, num_topics=num_topics, num_iterations=num_iterations)

            p, r, f, _ = nmf_evaluator.get_docs_classification_performance(run['theta'], dataset,
                                                                           num_folds=num_folds, random_seed=i)
            if prec is None:
                prec, rec, f1 = [], [], []
            prec.append(p[:, 1].mean())
            rec.append(r[:, 1].mean())
            f1.append(f[:, 1].mean())

        elif model in {'NVDM', 'GSM', 'NVLDA', 'ProdLDA', 'Scholar', 'NSMDM', 'NSMTM', 'NVCTM'}:
            params = neural_trainer.default_params(model=model, num_epoch=num_iterations, gpu_index=gpu_index)
            run = neural_trainer.train(dataset=dataset, dataset_name='None', num_topics=num_topics, params=params)

            result = neural_evaluator.get_docs_classification_performance(model_name=model, model=run['model'],
                                                                          dataset=dataset, num_folds=num_folds,
                                                                          random_seed=i)
            if prec is None:
                if result['num_options'] > 1:
                    prec = [[] for _ in range(result['num_options'])]
                    rec = [[] for _ in range(result['num_options'])]
                    f1 = [[] for _ in range(result['num_options'])]
                else:
                    prec, rec, f1 = [], [], []
            if result['num_options'] > 1:
                for r in range(result['num_options']):
                    prec[r].append(result['performance'][r][0][:, 1].mean())
                    rec[r].append(result['performance'][r][1][:, 1].mean())
                    f1[r].append(result['performance'][r][2][:, 1].mean())
            else:
                prec.append(result['performance'][0][:, 1].mean())
                rec.append(result['performance'][1][:, 1].mean())
                f1.append(result['performance'][2][:, 1].mean())

        else:
            print('doc classification is not defined for model {}'.format(model))
    return prec, rec, f1


def create_options():
    models = ['online_lda',
              'gibbs_lda',
              # 'biterm_lda',
              'nmf',
              'NVDM',
              'GSM',
              'NVLDA',
              'ProdLDA',
              'Scholar',
              'NSMDM',
              'NSMTM',
              'NVCTM']

    datasets = ['w2e',
                '20news',
                'w2e_text',
                'snippets']

    num_topics = {'w2e': [15, 30, 45, 60, 75, 90],
                  '20news': [10, 20, 30, 40, 50, 60],
                  'w2e_text': [15, 30, 45, 60, 75, 90],
                  'snippets': [4, 8, 12, 16, 20, 24]
                  }
    options = []
    run_no = 0
    for d in range(len(datasets)):
        for n in num_topics[datasets[d]]:
            for model in models:
                gpu_index = run_no % 13 + 3
                options.append({'run': run_no, 'dataset': datasets[d],
                                'num_topics': n, 'model': model, 'gpu_index': gpu_index})
                run_no += 1
    return options


def examine(option):
    data_file = 'preprocessed_data/{}_preprocessed_data.txt'.format(option['dataset'])
    dataset = Dataset(data_file, sentence_list=True, word_list=True, num_sentences=True, num_words=True,
                      stopword_label=True, raw_text=True, label=True)
    model = option['model']
    n = option['num_topics']
    gpu_index = option['gpu_index']
    doc_perplexity = unseen_doc_perplexity(model, dataset, n, gpu_index=gpu_index)
    print('model = {} num_topics = {} doc_perplexity = {}'.format(model, n, doc_perplexity))

    word_perplexity = unseen_words_perplexity(model, dataset, n, gpu_index=gpu_index)
    print('model = {} num_topics = {} word_perplexity = {}'.format(model, n, word_perplexity))

    coherence = topic_coherence(model, dataset, n, gpu_index=gpu_index)
    print('model = {} num_topics = {} coherence = {}'.format(model, n, coherence))

    classification_performance = doc_classification(model, dataset, n, gpu_index=gpu_index)
    print('model = {} num_topics = {} classification_performance = {}'.format(model, n,
                                                                              classification_performance))
    run = {'option': option,
           'doc_perplexity': doc_perplexity,
           'word_perplexity': word_perplexity,
           'coherence': coherence,
           'classification_performance': classification_performance}
    pickle.dump(run, open('result/run.{}.pkl'.format(option['run']), 'wb'))


def run_experiment(num_jobs=10):
    options = create_options()
    for option in options:
        print(option)
    Parallel(n_jobs=num_jobs)(delayed(examine)(option) for option in options)
