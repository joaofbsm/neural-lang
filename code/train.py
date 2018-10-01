#!/usr/bin/env python3

"""Training of Word2Vec Neural Language Models for Word Prediction"""

__author__ = 'João Francisco Barreto da Silva Martins'
__email__ = 'joaofbsm@dcc.ufmg.br'
__license__ = 'MIT'

import sys
import logging
import nlp
import gensim


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


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
    for corpus_proportion in corpus_proportions:
        sentences = gensim.models.word2vec.LineSentence(
                '{}{}{}'.format(corpus_path, corpus_proportion, corpus_filename)
        )

        for context_size in context_sizes:
            for algorithm_name, sg in training_algorithms.items():
                model = gensim.models.Word2Vec(
                        sentences=sentences,
                        window=context_size,
                        min_count=1,
                        workers=4,
                        sg=sg
                )

                model.save('{}{}-{}-{}.model'.format(
                        models_path,
                        corpus_proportion,
                        context_size,
                        algorithm_name)
                )


if __name__ == '__main__':
    main(sys.argv[1:])
