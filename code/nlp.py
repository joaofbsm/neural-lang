#!/usr/bin/env python3

"""Creation and Evaluation of Word2Vec Neural Language Models"""

__author__ = 'João Francisco Barreto da Silva Martins'
__email__ = 'joaofbsm@dcc.ufmg.br'
__license__ = 'MIT'

import logging
import gensim
import numpy as np


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


def prepare_validation_file(path_to_file='', filename='validation.txt',
                            prefix_filter=None, lowercase=False):
    """
    Prepare the validation file for evaluation of analogy similarity.
    All actions are optional.

    :param path_to_file: path to the validation file.
    :param filename: validation file name.
    :param prefix_filter: prefix substring to filter line out of file.
    :param lowercase: flag to convert all words in file to lowercase.
    :return: None
    """

    # Removes topic's headers
    if prefix_filter is not None:
        with open('{}{}'.format(path_to_file, filename), 'r') as old_file, \
             open('{}prep_{}'.format(path_to_file, filename), 'w+') as new_file:

            for line in old_file:
                if not line.startswith(prefix_filter):

                    # Convert all words to lowercase
                    if lowercase:
                        line = line.lower()

                    new_file.write(line)

    # Convert all words to lowercase in case there is no line filter
    elif lowercase:
        with open('{}{}'.format(path_to_file, filename), 'r') as old_file, \
             open('{}prep_{}'.format(path_to_file, filename), 'w+') as new_file:

            for line in old_file:
                line = line.lower()
                new_file.write(line)


def evaluate_analogies_distance(model, validation_path, validation_filename):
    """

    :param model:
    :param validation_path:
    :param validation_filename:
    :return:
    """

    oov_question = 0
    oov_answer = 0
    distances = []

    with open(validation_path + validation_filename, 'r') as f:
        for line in f:
            words = line.split()

            # Get word on top of the similarity to the resulting vector rank
            try:
                predicted = model.most_similar(positive=words[1:3],
                                               negative=words[0],
                                               topn=1)[0][0]
            except:
                oov_question += 1
                continue
            # Calculate the distance between predicted and correct word
            try:
                distances.append(float(model.wv.distance(predicted, words[3])))
            except:
                oov_answer += 1
                continue

    mean_distances = np.mean(distances)

    logging.info('Mean of analogies distance: {}'.format(mean_distances))

    logging.info(('{} question words and {} answer words out of '
                  'vocabulary').format(oov_question, oov_answer))

    return mean_distances


def train_models(corpus_path, corpus_filename, models_path, corpus_proportions,
                 context_sizes, training_algorithms):
    """

    :param corpus_path:
    :param corpus_filename:
    :param models_path:
    :param corpus_proportions:
    :param context_sizes:
    :param training_algorithms:
    :return:
    """

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


def test_models(validation_path, validation_filename, models_path, results_path,
                corpus_proportions, context_sizes, training_algorithms):
    """

    :param validation_path:
    :param validation_filename:
    :param models_path:
    :param results_path:
    :param corpus_proportions:
    :param context_sizes:
    :param training_algorithms:
    :return:
    """

    for corpus_proportion in corpus_proportions:
        for context_size in context_sizes:
            for algorithm_name, sg in training_algorithms.items():
                model = gensim.models.Word2Vec.load('{}{}-{}-{}.model'.format(
                        models_path,
                        corpus_proportion,
                        context_size,
                        algorithm_name)
                )

                accuracy = model.wv.evaluate_word_analogies(
                        validation_path + validation_filename,
                        case_insensitive=True
                )[0]

                distance = evaluate_analogies_distance(
                        model,
                        validation_path,
                        'prep_' + validation_filename
                )

                with open('{}{}-{}-{}.txt'.format(
                        results_path,
                        corpus_proportion,
                        context_size,
                        algorithm_name),
                           'w+') as f:
                    f.write('accuracy={:.5g}\n'.format(accuracy))
                    f.write('distance={:.5g}'.format(distance))
