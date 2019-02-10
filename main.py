#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn import tree, preprocessing, metrics, impute
from sklearn.model_selection import cross_val_score

from util import load_data_set


def decision_tree_pruning(options):
    car_data = load_data_set('car')
    car_ohe = preprocessing.OneHotEncoder()
    car_ohe.fit(car_data['train']['inputs'] + car_data['test']['inputs'])  # encode features as one-hot

    cancer_data = load_data_set('breastcancer')
    cancer_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    cancer_imp.fit(np.array(cancer_data['train']['inputs'] + cancer_data['test']['inputs'], dtype=np.float32))

    x = list()
    y_f1_test = list()
    y_f1_train = list()
    y_cross = list()

    for i in range(20):  # max depth from 1 to 39 in steps of 2
        max_depth = 1 + 2 * i
        x.append(max_depth)

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="random",
            min_samples_leaf=10,  # minimum of 10 samples at leaf nodes
            max_depth=max_depth
        )
        clf.fit(car_ohe.transform(car_data['train']['inputs']), car_data['train']['outputs'])

        predicted = clf.predict(car_ohe.transform(car_data['train']['inputs']))
        train_f1_score = metrics.f1_score(car_data['train']['outputs'], predicted, average='micro')
        predicted = clf.predict(car_ohe.transform(car_data['test']['inputs']))
        test_f1_score = metrics.f1_score(car_data['test']['outputs'], predicted, average='micro')

        data_in = car_ohe.transform(car_data['train']['inputs'] + car_data['test']['inputs'])
        data_out = car_data['train']['outputs'] + car_data['test']['outputs']

        y_f1_train.append(train_f1_score)
        y_f1_test.append(test_f1_score)
        y_cross.append(np.mean(cross_val_score(clf, data_in, data_out, cv=5)))

    plt.figure()
    plt.plot(x, y_f1_train, label='training F1 score')
    plt.plot(x, y_f1_test, label='test F1 score')
    plt.plot(x, y_cross, label='cross-validation')

    plt.title('Max depth v. decision tree performance (car.dataset)')
    plt.xlabel('Max depth')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('out/decision_tree_pruning/car-maxdepth.png', dpi=300)

    x = list()
    y_f1_test = list()
    y_f1_train = list()
    y_cross = list()

    for i in range(20):  # max depth from 1 to 39 in steps of 2
        max_depth = 1 + 2 * i
        x.append(max_depth)

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="random",
            min_samples_leaf=10,  # minimum of 10 samples at leaf nodes
            max_depth=max_depth
        )
        clf.fit(cancer_imp.transform(cancer_data['train']['inputs']), cancer_data['train']['outputs'])

        predicted = clf.predict(cancer_imp.transform(cancer_data['train']['inputs']))
        train_f1_score = metrics.f1_score(cancer_data['train']['outputs'], predicted, average='micro')
        predicted = clf.predict(cancer_imp.transform(cancer_data['test']['inputs']))
        test_f1_score = metrics.f1_score(cancer_data['test']['outputs'], predicted, average='micro')

        data_in = cancer_imp.transform(cancer_data['train']['inputs'] + cancer_data['test']['inputs'])
        data_out = cancer_data['train']['outputs'] + cancer_data['test']['outputs']

        y_f1_train.append(train_f1_score)
        y_f1_test.append(test_f1_score)
        y_cross.append(np.mean(cross_val_score(clf, data_in, data_out, cv=5)))

        skplt.estimators.plot_learning_curve(
            clf, data_in, data_out, title="Learning Curve: Decision Trees (breastcancer.dataset)")

    plt.figure()
    plt.plot(x, y_f1_train, label='training F1 score')
    plt.plot(x, y_f1_test, label='test F1 score')
    plt.plot(x, y_cross, label='cross-validation')

    plt.title('Max depth v. decision tree performance (breastcancer.dataset)')
    plt.xlabel('Max depth')
    plt.ylabel('Score!')
    plt.legend()
    plt.savefig('out/decision_tree_pruning/breastcancer-maxdepth.png', dpi=300)

    plt.show()
    print("done")


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
