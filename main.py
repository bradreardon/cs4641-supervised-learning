#!/usr/bin/env python3
import argparse

from algorithms.boosting import boosting
from algorithms.decision_tree_pruning import decision_tree_pruning
from algorithms.knn import knn
from algorithms.neural_net import neural_net
from algorithms.svm import svm_


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
    parser_svm.set_defaults(func=svm_)
    parser_svm.add_argument(
        '-k', '--kernel', default='linear',
        help="Selects the kernel to use with SVM. Options: linear, rbf, poly. Defaults to linear.")

    # k-nearest neighbors parser
    parser_knn = subparsers.add_parser(
        'knn', help='Runs supervised learning using k-nearest neighbors.')
    parser_knn.set_defaults(func=knn)
    parser_knn.add_argument(
        '-k', '--k_value', type=int, default=1,
        help="Sets the value k. Defaults to 1.")

    # Parse args and jump into correct algorithm
    options = parser.parse_args()
    options.func(options)
