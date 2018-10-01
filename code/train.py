#!/usr/bin/env python3

"""Training of Word2Vec Neural Language Models for Word Analogy"""

__author__ = 'João Francisco Barreto da Silva Martins'
__email__ = 'joaofbsm@dcc.ufmg.br'
__license__ = 'MIT'

import sys
import nlp


def main(args):
    # Arguments
    corpus_filename = args[0].split('/')[-1]
    corpus_path = args[0][:-len(corpus_filename)]
    models_path = args[1]

    # Execution parameters
    corpus_proportions = (0.25, 0.5, 0.75, 1)
    context_sizes = (5, 10, 20, 100)
    training_algorithms = {'cbow': 0, 'sg': 1}  # 0 for CBOW, 1 for Skip-Gram

    # Prepare corpus for training
    nlp.split_corpus_file(corpus_path, corpus_filename, corpus_proportions)

    # Train models with a varied set of parameters
    nlp.train_models(corpus_filename, corpus_path, models_path,
                     corpus_proportions, context_sizes, training_algorithms)


if __name__ == '__main__':
    main(sys.argv[1:])
