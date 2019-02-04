#!/usr/bin/env python3.7
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, preprocessing, metrics

from datasets.util import split_data


NUM_TRIALS = 20


def decision_tree_pruning(options):
    x_splits = list()
    y_f1_test = list()
    y_f1_train = list()

    for i in range(10):
        split_pct = (i + 1) * .05  # 5% steps
        x_splits.append(split_pct)
        trains = list()
        tests = list()

        for j in range(NUM_TRIALS):
            with open('./datasets/car/car.data') as f:
                d = [line.strip().split(',') for line in f.readlines()]
                d = [
                    (sample[0:-1], [sample[-1]]) for sample in d
                ]

                train, test = split_data(d, split_pct=split_pct)
                train_data_in, train_data_out = [s[0] for s in train], [s[1] for s in train]
                test_data_in, test_data_out = [s[0] for s in test], [s[1] for s in test]

                ohe = preprocessing.OneHotEncoder()

                ohe.fit(train_data_in)  # encode features as one-hot

            # print(f"decision_tree_pruning with ./datasets/car, split_pct={split_pct:.2f}")

            # set up classifier to limit number of leaves
            clf = tree.DecisionTreeClassifier(
                # criterion="gini",
                # splitter='random',
                min_samples_leaf=3,  # minimum of X samples at leaf nodes
                max_depth=10
            )
            clf.fit(ohe.transform(train_data_in), train_data_out)

            # score = clf.score(ohe.transform(test_data_in), test_data_out)
            # print(f"score: {score}")

            predicted = clf.predict(ohe.transform(train_data_in))
            train_f1_score = metrics.f1_score(train_data_out, predicted, average='micro')
            predicted = clf.predict(ohe.transform(test_data_in))
            test_f1_score = metrics.f1_score(test_data_out, predicted, average='micro')

            trains.append(train_f1_score)
            tests.append(test_f1_score)

        y_f1_train.append(np.mean(trains))
        y_f1_test.append(np.mean(tests))

    plt.plot(x_splits, y_f1_train, label='training data')
    plt.plot(x_splits, y_f1_test, label='test data')

    plt.xlabel('% of data in test set')
    plt.ylabel('F1 score')
    plt.show()


def neural_net(options):
    print("neural_net")


def boosting(options):
    print("boosting")


def svm(options):
    kernel = options.kernel
    print(f"svm, kernel: {kernel}")


def k_nearest(options):
    k = options.k_value
    print(f"k_nearest, k={k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs supervised learning.")
    subparsers = parser.add_subparsers()
    parser.set_defaults(func=lambda x: parser.print_help())

    # Decision tree parser
    parser_decision_tree_pruning = subparsers.add_parser(
        'decision_tree_pruning', help='Runs supervised learning using decision trees, with pruning.')
    parser_decision_tree_pruning.set_defaults(func=decision_tree_pruning)

    # Neural net parser
    parser_neural_net = subparsers.add_parser(
        'neural_net', help='Runs supervised learning using neural networks.')
    parser_neural_net.set_defaults(func=neural_net)

    # Boosting parser
    parser_boosting = subparsers.add_parser(
        'boosting', help='Runs supervised learning using boosting.')
    parser_boosting.set_defaults(func=boosting)

    # SVM Parser
    parser_svm = subparsers.add_parser(
        'svm', help='Runs supervised learning using Support Vector Machines.')
    parser_svm.set_defaults(func=svm)
    parser_svm.add_argument(
        '-k', '--kernel', default='todo1',
        help="Selects the kernel to use with SVM. Options: todo1, todo2. Defaults to todo1.")

    # k-nearest neighbors parser
    parser_k_nearest = subparsers.add_parser(
        'k_nearest', help='Runs supervised learning using k-nearest neighbors.')
    parser_k_nearest.set_defaults(func=k_nearest)
    parser_k_nearest.add_argument(
        '-k', '--k_value', type=int, default=1,
        help="Sets the value k. Defaults to 1.")

    # Parse args and jump into correct algorithm
    options = parser.parse_args()
    options.func(options)
