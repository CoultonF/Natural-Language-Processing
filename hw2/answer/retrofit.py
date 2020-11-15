import math
import numpy
import re
import sys
from pymagnitude import *
from copy import deepcopy


def print_output(word_vecs, outFile):
    w_file = open(outFile,'w')
    for word in wordvecs:
        w_file.write(word)
        for vec in word_vecs[word]:
            w_file.write(' '+str(vec))
        w_file.write('\n')
    w_file.close()

def read_wordvecs(filename):
    wordvecs = {}
    wv = Magnitude(filename)
    
    for word in wv:
        wordvecs[word[0]] = word[1]
    return wordvecs 

def lex_dict(filename):
        lexicon = {}
        for line in open(filename,'r'):
            word_list = line.lower().strip().split()
            lexicon[word_list[0]] = [word for word in word_list[1:]]
        return lexicon


def retrofit(wordvecs, lexicon, iter):
    new_vecs = deepcopy(wordvecs)
    vocab = set(new_vecs.keys())
    common_vocab = vocab.intersection(set(lexicon.keys()))
    for i in range(iter):
        for word in common_vocab:
            w_neighbours = set(lexicon[word]).intersection(vocab)
            n_neighbours = len(w_neighbours)
            if n_neighbours > 0:
                new_vector = n_neighbours * wordvecs[word]
                for pword in w_neighbours:
                    new_vector = new_vector + new_vecs[pword]
                new_vecs[word] = new_vector/(2*n_neighbours)
    return new_vecs

wordvecs = read_wordvecs('data/glove.6B.100d.magnitude')
lexicon = lex_dict('data/lexicons/wordnet-synonyms.txt')
print_output(retrofit(wordvecs,lexicon,10),'data/glove.6B.100d.retrofit.txt')

