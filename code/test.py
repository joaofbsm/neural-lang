#!/usr/bin/env python3

"""Test of Word2Vec Neural Language Models for Word Prediction"""

__author__ = 'João Francisco Barreto da Silva Martins'
__email__ = 'joaofbsm@dcc.ufmg.br'
__license__ = 'MIT'

import sys
import logging
import code.nlp as nlp
import gensim


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def main(args):
    # Arguments
    corpus_filename = args[0]
    validation_filename = args[1].split('/')[-1]
    validation_path = args[1][:-len(validation_filename)]
    models_path = args[2]

    # Execution parameters
    corpus_proportions = (0.25, 0.5, 0.75, 1)
    context_sizes = (5, 10, 20, 100)
    training_algorithms = {'cbow': 0, 'sg': 1}  # 0 for CBOW, 1 for Skip-Gram

    nlp.prepare_validation_file(validation_path, validation_filename,
                                prefix_filter=':', lowercase=True)


if __name__ == '__main__':
    main(sys.argv[1:])
