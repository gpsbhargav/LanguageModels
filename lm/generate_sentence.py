
# coding: utf-8

import os
from random import shuffle
from random import randint
from random import uniform
from math import floor
import shutil
import codecs
import math
from collections import Counter
from collections import defaultdict

import numpy as np

import nltk
from nltk import word_tokenize
from nltk.util import ngrams

#brown train and test
#TRAIN_PATH = '../data/brown/train/train_and_validate.txt'
#TEST_PATH = '../data/brown/test/test.txt'

#gutenberg train and test
TRAIN_PATH = '../data/gutenberg/train/train_and_validate.txt'
#TEST_PATH = '../data/gutenberg/test/test.txt'

#both train, brown test
#TRAIN_PATH = '../data/both_train/both_train.txt'
#TEST_PATH = '../data/brown/test/test.txt'

#both train, gutenberg test
#TRAIN_PATH = '/home1/e1-246-65/assignment1/data/both_train/both_train.txt'
#TEST_PATH = '../data/gutenberg/test/test.txt'


class KNLM:
    def __init__(self,order=2):
        self.order = 2
    
    def set_training_data(self,train_file):
        '''
        reads file, tokenizes, replaces "some" rare words with <unk>. stores these in the instance 
        variable tokens
        '''
        f = codecs.open(train_file, encoding='utf-8')
        tokens = nltk.word_tokenize(f.read())
        f.close()
        tokens_original = tokens
        unigram_counter = Counter(ngrams(tokens,1))        
        num_unk = 0
        for i in range(len(tokens_original)):
            if(  unigram_counter[tuple([tokens_original[i]])] < 3):
                tokens[i] = '<unk>'
                num_unk = num_unk + 1

        self.tokens = ["<pad>"] + tokens
        self.unigram_counter = Counter(ngrams(self.tokens,1))
        self.bigram_counter = Counter(ngrams(self.tokens,2))
        self.vocabulary = set(self.tokens)
        self.vocabulary_list = list(self.vocabulary)
        
    def train(self):
        self.unique_continuations = defaultdict(set)
        self.unique_contexts = defaultdict(set)
        for bigram in self.bigram_counter.keys():
            self.unique_continuations[bigram[0]].add(bigram[1])
            self.unique_contexts[bigram[1]].add(bigram[0])
    
    def get_prob_kn(self,ngram):
        abs_discount = 0.85
        count_input_ngram = max(self.bigram_counter[ngram] - abs_discount , 0)
        count_context = self.unigram_counter[tuple([ngram[0]])]
        num_unique_continuations = len(self.unique_continuations[ngram[0]])
        num_unique_contexts = len(self.unique_continuations[ngram[1]])
        interpolation_weight = (abs_discount / count_context) * num_unique_continuations
        continuation_probability = num_unique_contexts / len(self.bigram_counter)
        p_kn = (count_input_ngram / count_context) + (interpolation_weight * continuation_probability)
        return p_kn
    
    def __preproc_test_input(self,tokens):
        '''converts words not in vocab to <unk>'''
        unk_count = 0
        for i in range(len(tokens)):
            if(tokens[i] not in  self.vocabulary):
                tokens[i] = '<unk>'
                unk_count = unk_count + 1
        tokens = ['<pad>'] + tokens
        return tokens
    
    def get_perplexity(self,test_file):
        f = codecs.open(test_file, encoding='utf-8')
        tokens = nltk.word_tokenize(f.read())
        tokens = self.__preproc_test_input(tokens)
        f.close()
        bigrams = nltk.ngrams(tokens,2)
        sum_log_prob = 0
        iter_count = 0
        for bigram in bigrams: 
            prob = math.log(self.get_prob_kn(bigram))
            sum_log_prob = sum_log_prob + prob
            iter_count += 1
        return math.exp( -(1.0/len(tokens)) * sum_log_prob )
    
    def get_random_word(self):
        index = randint(0,len(self.vocabulary_list)-1)
        return kn_lm.vocabulary_list[index]

    
    def generate_text(self,num_tokens=10):
        sentence = ['<pad>']
        while(len(sentence) < num_tokens+1):
            random_word = self.get_random_word()
            if(random_word == '<unk>'):
                continue
            bigram = tuple(sentence[-1:] + [random_word])
            prob_bigram = self.get_prob_kn(bigram)
            random_number = uniform(0,1)
            if(random_number < prob_bigram):
                sentence = sentence + [random_word]
        print(' '.join(sentence[1:]))
    

kn_lm = KNLM()

kn_lm.set_training_data(TRAIN_PATH)

kn_lm.train()

#print(kn_lm.get_perplexity(TEST_PATH))

kn_lm.generate_text()
