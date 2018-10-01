#!/usr/bin/env python3

"""Test of Word2Vec Neural Language Models for Word Analogy"""

__author__ = 'João Francisco Barreto da Silva Martins'
__email__ = 'joaofbsm@dcc.ufmg.br'
__license__ = 'MIT'

import sys
import nlp


def main(args):
    # Arguments
    validation_filename = args[0].split('/')[-1]
    validation_path = args[0][:-len(validation_filename)]
    models_path = args[1]
    results_path = args[2]

    # Execution parameters
    corpus_proportions = (0.25, 0.5, 0.75, 1)
    context_sizes = (5, 10, 20, 100)
    training_algorithms = {'cbow': 0, 'sg': 1}  # 0 for CBOW, 1 for Skip-Gram

    # Prepare validation file for testing
    nlp.prepare_validation_file(validation_path, validation_filename,
                                prefix_filter=':', lowercase=True)

    # Evaluate the models in terms of accuracy and the proposed distance metric
    nlp.test_models(validation_path, validation_filename,  models_path,
                    results_path, corpus_proportions, context_sizes,
                    training_algorithms)


if __name__ == '__main__':
    main(sys.argv[1:])
