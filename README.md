# Examined models

## Neural models
- [x] NVDM from https://arxiv.org/pdf/1511.06038.pdf -- ICML 2016
- [x] GSM from https://arxiv.org/pdf/1706.00359.pdf -- ICML 2017
- [x] NVLDA from https://arxiv.org/pdf/1703.01488.pdf -- ICLR 2017
- [x] ProdLDA from https://arxiv.org/pdf/1703.01488.pdf -- ICLR 2017
- [x] NSMTM from https://arxiv.org/pdf/1810.09079.pdf -- WSDM 2019
- [x] NSMDM from https://arxiv.org/pdf/1810.09079.pdf -- WSDM 2019
- [x] Scholar from https://arxiv.org/abs/1705.09296 -- ACL 2018
- [x] NVCTM from https://dl.acm.org/doi/10.1145/3308558.3313561 -- WWW 2019


## Non-neural models
- [x] online-LDA: LDA using online variational inference
- [x] online-LDA: LDA using Gibbs sampling
- [x] NMF: [online NMF](http://proceedings.mlr.press/v54/zhao17a.html)


# Metrics:
- Perplexity of unseen documents: All models, except LDA_gibbs and NMF
- Perplexity of unseen/held-out words: All models, except NMF
- Topic coherence: All models
- Performance in document classification: All models



# Datasets:
## Short texts:
* [1] Title of news articles in W2E dataset from https://dl.acm.org/doi/abs/10.1145/3269206.3269309
* [2] Web snippets from https://papers.nips.cc/paper/2002/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html 

## Long texts:
* [1] Content of news articles in W2E dataset from https://dl.acm.org/doi/abs/10.1145/3269206.3269309
* [2] 20News

The datasets are preprocessed and shared [here](https://drive.google.com/drive/folders/12t2fLjkswmcogdKYTCEKNLR3bQsatB5-?usp=sharing). 
Please download and unzip the files into the preprocessed_data folder

# Script for training neural topic models:
## Trainers:
- train/trainer_neural_topic_model.py: for neural models
- train/trainer_lda_topic_model.py: for LDA models
- train/trainer_nmf_topic_model.py: for NMF models

## Evaluators:
- evaluation/eval_neural_models.py: for neural models
- evaluation/eval_lda_models.py: for LDA models
- evaluation/eval_nmf_models.py: for NMF models

## Running experiments:
- empirical_studies/examine_models.py