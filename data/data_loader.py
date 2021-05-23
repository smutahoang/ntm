import operator
import json
import random


class Dataset:
    """
    """

    def __init__(self, preprocessed_data_file,
                 sentence_list=True,
                 word_list=False,
                 bow=True,
                 num_sentences=False,
                 num_words=True,
                 stopword_label=False,
                 raw_text=False,
                 label=False):
        """

        :param preprocessed_data_file:
        :param sentence_list: if list of sentences is necessary for the learning
        :param word_list: if list of words is necessary for the learning
        :param num_sentences: if #sentences is necessary for the learning
        :param num_words: if #words is necessary for the learning
        :param stopword_label:
        """
        self.vocab = []
        self.word2index = dict()
        self.stopword_indexes = set()
        self.word_frequency = []

        self.PAD_WORD = 'PAD_WORD'
        self.word2index[self.PAD_WORD] = len(self.vocab)
        self.stopword_indexes.add(len(self.vocab))
        self.vocab.append(self.PAD_WORD)
        self.word_frequency.append(0)

        self.EOS_MARKER = 'EOS_MARKER'
        self.word2index[self.EOS_MARKER] = len(self.vocab)
        self.stopword_indexes.add(len(self.vocab))
        self.vocab.append(self.EOS_MARKER)
        self.word_frequency.append(0)

        self.documents = []
        self.categories = dict()

        self.word_sampler = None  # for quick sampling of words with probability proportional to their weight

        file = open(preprocessed_data_file, 'r')
        for line in file:
            line = line.strip()
            doc = json.loads(line)
            #
            fields = dict()
            fields['id'] = doc['id']
            #
            sentences = []
            nwords = 0
            for s in doc['sentences']:
                sentence = []
                words = s.split()
                for w in words:
                    if w in self.word2index:
                        index = self.word2index[w]
                        sentence.append(index)
                        self.word_frequency[index] += 1
                    else:
                        index = len(self.vocab)
                        self.word2index[w] = index
                        sentence.append(index)
                        self.stopword_indexes.add(index)
                        self.word_frequency.append(1)
                        self.vocab.append(w)
                sentences.append(sentence)
                nwords += len(sentence)
            if sentence_list:
                fields['sentence_list'] = sentences

            #
            if word_list:
                words = []
                for s in sentences:
                    words += s
                fields['word_list'] = words

            #
            bag_of_words = []
            for w in doc['bow'].split():
                index = self.word2index[w]
                bag_of_words.append(index)
                self.stopword_indexes.discard(index)
            if bow:
                fields['bow'] = bag_of_words

            #
            if stopword_label:
                unique_words = set(bag_of_words)
                stopword_labels = []
                for sentence in sentences:
                    swlabel = []
                    for w in sentence:
                        if w in unique_words:
                            swlabel.append(0)
                        else:
                            swlabel.append(1)
                    stopword_labels += swlabel
                fields['stopword_label'] = stopword_labels

            #
            if num_sentences:
                fields['num_sentences'] = len(sentences)

            #
            if num_words:
                fields['num_words'] = nwords

            #
            if raw_text:
                fields['raw_text'] = doc['sentences']

            #
            if label:
                labels = []
                for l in doc['categories'].split():
                    if l in self.categories:
                        labels.append(self.categories[l])
                    else:
                        labels.append(len(self.categories))
                        self.categories[l] = len(self.categories)
                fields['labels'] = labels

            #
            self.documents.append(fields)

        self.doc_of_word = [[] for i in range(len(self.vocab))]
        for docId in range(len(self.documents)):
            fields = self.documents[docId]
            doc = fields['sentence_list'][0]  # TODO 0 because each doc has one sentence only
            for wId in doc:
                self.doc_of_word[wId].append(docId)

        self.print_info()

    #
    def print_info(self):
        print('#docs = ', len(self.documents))
        print('vocab_size = ', len(self.vocab))

    #
    def batching(self, batch_size, evaluate=False):
        indexes = [i for i in range(len(self.documents))]
        if not evaluate:
            random.shuffle(indexes)
        batches = []
        n = len(indexes) // batch_size
        for b in range(n):
            batches.append(indexes[b * batch_size:(b + 1) * batch_size])
        # last batch:
        if b * n < len(indexes):
            if evaluate:
                batches.append(indexes[batch_size * n:])
            else:
                batches.append(indexes[-batch_size:])
        return batches

    #
    def get_data_batch(self, batch_indexes):
        doc_batch = [self.documents[index] for index in batch_indexes]
        return doc_batch

    def get_word_batch(self, batch_indexes):
        word_batch = [self.doc_of_word[ind] for ind in batch_indexes]
        return word_batch

    def get_word_doc_freq(self):
        counts = dict()
        for doc in self.documents:
            unique_words = list(set(doc['bow']))
            for i in range(len(unique_words)):
                w = unique_words[i]
                if w not in counts:
                    counts[w] = 1
                else:
                    counts[w] += 1
        sorted_words = sorted(counts.items(), key=operator.itemgetter(1))
        return sorted_words

    def get_top_words(self, label, topk):
        counts = dict()
        for doc in self.documents:
            if not label in doc['labels']:
                continue
            unique_words = list(set(doc['bow']))
            for i in range(len(unique_words)):
                w = unique_words[i]
                if w not in counts:
                    counts[w] = 1
                else:
                    counts[w] += 1
        sorted_words = sorted(counts.items(), key=operator.itemgetter(1))
        return sorted_words[:topk]

    def init_word_sampler(self):
        weight = [0] * len(self.vocab)
        for d in self.documents:
            for w in d['bow']:
                weight[w] += 1
        weight = [weight[i] ** (3 / 4) for i in range(len(self.vocab))]
        sum_weight = sum(weight)
        weight = [int(10E6 * weight[i] / sum_weight) for i in range(len(self.vocab))]
        self.word_sampler = []
        for i in range(len(self.vocab)):
            if weight[i] > 0:
                self.word_sampler += [i] * weight[i]

    def quick_word_sample(self, num_words):
        if self.word_sampler is None:
            self.init_word_sampler()
        samples = [self.word_sampler[random.randint(0, len(self.word_sampler) - 1)] for i in range(num_words)]
        return samples

    def negative_sampling(self, d, num_words):
        true_words = set(self.documents[d]['bow'])
        negative_samples = []
        while num_words > 0:
            samples = self.quick_word_sample(num_words)
            samples = [w for w in samples if w not in true_words]
            negative_samples += samples
            num_words -= len(negative_samples)
        return negative_samples

    def split_bow(self, random_seed, ratio=0.1):
        """
        to split the bag_of_words into training & and testing data by test
        """
        random.seed(random_seed)
        for document in self.documents:
            num_test_words = max(1, int(ratio * len(document['bow'])))
            test_words = random.sample(document['bow'], num_test_words)
            for w in test_words:
                document['bow'].remove(w)
            document['test_bow'] = test_words

    def get_list_sentences(self):
        """
        This is used for word2vec
        """
        sentences = []
        count = 0
        for fields in self.documents:
            sentence_list = fields['sentence_list']
            if len(sentence_list) > 1:
                count += 1
            sent = [self.vocab[i] for i in sentence_list[0]]
            sentences.append(sent)  # since no sentence_list has more than 2 elements
        return sentences

# dataset = Dataset('/home/hoang/mnt/data/arxiv_preprocessed_data.txt')
# print(dataset.vocab[:10])
