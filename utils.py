import os
import math
import torch
import heapq
import random
import pickle
import datetime
from rouge import rouge
from bleu import compute_bleu
import pandas as pd

def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class WordDictionary:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        if type(sentence)==float:
            print(sentence)
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)

# select longest aspect
select_longest = lambda triplets: max(triplets, key=lambda trip: len(trip[0]))


class DataLoader:
    def __init__(self, data_path, index_dir, vocab_size):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.train, self.valid, self.test = self.load_data(data_path, index_dir)
        # self.user_aspect_top2=torch.load('./user_aspect_top2.pt')

    def initialize(self, data_path):
        # assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:

            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            tem = review['template'][2]
            self.word_dict.add_sentence(tem)
            # self.word_dict.add_word(fea)
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            # (fea, adj, tem, sco, cat) = review['template']
            fea, adj, tem, sco, cat = select_longest(review['triplets'])
            # !!GET the longest!!
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'user_id': review['user'],
                        'item_id': review['item'],
                         'rating': review['rating'],
                         'text': tem,
                         "category": cat,
                         'feature': self.word_dict.word2idx.get(fea, self.__unk)
               })
            if fea in self.word_dict.word2idx:
                self.feature_set.add(fea)
            else:
                self.feature_set.add('<unk>')

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        for idx in train_index:
            train.append(data[idx])
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]

    def load_index(self, index_dir):

        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


def sentence_format(sentence, max_len, pad, bos, eos):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


class Batchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False,user_aspect=None):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        # for later use of generation prediction file
        self.users = []
        self.items = []
        u, i, r, t, f = [], [], [], [], []
        for x in data:

            self.users.append(x['user'])
            self.items.append(x['item'])
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(self.seq2ids(word2idx,x['text']), seq_len, pad, bos, eos))

            # fuid: factorized user id (0 - nuser-1)
            fuid = x['user']
            f.append(self.list2ids(word2idx,user_aspect[fuid])) # user_factorized_id:



        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.aspect = torch.tensor(f, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def seq2ids(self,word2idx, seq):
        self.__unk = word2idx['<unk>']
        return [word2idx.get(w, self.__unk) for w in seq.split()]



    def list2ids(self,word2idx, lst):
        """ lst: a list of strings (top 2 aspect terms)"""
        def get_aspect_word(word2idx,word):
            # check if the word is in the word2idx
            if word in word2idx:
                return word2idx[word]
            # otherwise we try to get the word by splitting it
            words=word.split()
            # if the word is not splittable, we return the unk token
            if len(words)==1:
                return word2idx['<unk>']
            else:
                for w in words[::-1]: # try from the last word to the first word
                    if w in word2idx:
                        return word2idx[w]
                return word2idx['<unk>']

        self.__unk = word2idx['<unk>']
        if len(lst)==1:
            tem1=[word2idx.get(w, self.__unk) for w in lst]
            tem1=tem1[0]
            return [tem1,tem1]
        if len(lst)==0:
            return [self.__unk,self.__unk]
        # he simply treat one aspect as one word
        temp=[get_aspect_word(word2idx,w) for w in lst]
        assert len(temp)==2
        return temp

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        aspect = self.aspect[index]  # (batch_size, 1)
        return user, item, rating, seq,aspect


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens
