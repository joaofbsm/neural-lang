#!/usr/bin/env python3

"""Creation and Evaluation of Word2Vec Neural Language Models"""

__author__ = 'João Francisco Barreto da Silva Martins'
__email__ = 'joaofbsm@dcc.ufmg.br'
__license__ = 'MIT'

import logging
import cython
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def split_corpus_file(path_to_file='', filename='corpus.txt',
                      proportions=(0.25, 0.5, 0.75, 1)):
    """
    Split corpus into different sizes.

    :param path_to_file: path to the corpus file.
    :param filename: corpus file name.
    :param proportions: proportions to apply to the size of corpus. One new file
                        for each value.
    :return: None
    """

    with open('{}{}'.format(path_to_file, filename), 'r') as f:
        # Considering that you are using a corpus with only one line
        corpus = f.read()

    corpus = corpus.split(' ')
    corpus_size = len(corpus)

    for proportion in proportions:
        splitting_point = round(corpus_size * proportion)
        with open('{}{}{}'.format(path_to_file, proportion, filename), 'w+') as f:
            f.write(' '.join(corpus[:splitting_point]))


def clean_validation_file(path_to_file='', filename='validation.txt',
                          prefix_filter=':'):
    """
    Removes topic header lines.

    :param path_to_file: path to the validation file.
    :param filename: validation file name.
    :param prefix_filter: prefix substring to filter line out of file.
    :return: None
    """

    with open('{}{}'.format(path_to_file, filename), 'r') as old_file, \
         open('{}clean_{}'.format(path_to_file, filename), 'w+') as new_file:

        for line in old_file:
            if not line.startswith(prefix_filter):
                new_file.write(line)


def evaluate_model():
    pass
