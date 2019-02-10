#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np
from sklearn import tree, svm, preprocessing, metrics, impute
from sklearn.metrics import accuracy_score, precision_score, log_loss

from util import load_data_set, export_decision_tree, Timer


def decision_tree_pruning_car():
    car_data = load_data_set('car')
    car_ohe = preprocessing.OneHotEncoder()
    car_ohe.fit(car_data['train']['inputs'] + car_data['test']['inputs'])  # encode features as one-hot

    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
    )

    with Timer() as t:
        clf.fit(car_ohe.transform(car_data['train']['inputs']), car_data['train']['outputs'])

    time_to_fit = t.interval * 1000

    predicted = clf.predict(car_ohe.transform(car_data['train']['inputs']))
    train_f1_score = metrics.f1_score(car_data['train']['outputs'], predicted, average='micro')

    with Timer() as t:
        predicted = clf.predict(car_ohe.transform(car_data['test']['inputs']))
    test_f1_score = metrics.f1_score(car_data['test']['outputs'], predicted, average='micro')

    test_prediction_runtime = t.interval * 1000

    data_in = car_ohe.transform(car_data['train']['inputs'] + car_data['test']['inputs'])
    data_out = car_data['train']['outputs'] + car_data['test']['outputs']

    t_out = car_data['test']['outputs']

    accuracy = accuracy_score(t_out, predicted) * 100
    precision = precision_score(t_out, predicted, average="weighted") * 100

    print("car.dataset (no pruning)")
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: Decision Trees (car.dataset, no pruning)", cv=5)
    plt.savefig('out/decision_tree_pruning/car-noprune-learning.png')
    export_decision_tree(clf, 'car-noprune')

    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
        min_samples_leaf=5,  # minimum of 5 samples at leaf nodes
        max_depth=9
    )

    with Timer() as t:
        clf.fit(car_ohe.transform(car_data['train']['inputs']), car_data['train']['outputs'])

    time_to_fit = t.interval * 1000

    predicted = clf.predict(car_ohe.transform(car_data['train']['inputs']))
    train_f1_score = metrics.f1_score(car_data['train']['outputs'], predicted, average='micro')

    with Timer() as t:
        predicted = clf.predict(car_ohe.transform(car_data['test']['inputs']))
    test_f1_score = metrics.f1_score(car_data['test']['outputs'], predicted, average='micro')

    test_prediction_runtime = t.interval * 1000

    data_in = car_ohe.transform(car_data['train']['inputs'] + car_data['test']['inputs'])
    data_out = car_data['train']['outputs'] + car_data['test']['outputs']

    t_out = car_data['test']['outputs']

    accuracy = accuracy_score(t_out, predicted) * 100
    precision = precision_score(t_out, predicted, average="weighted") * 100

    print("car.dataset (pruned)")
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()
    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: Decision Trees (car.dataset, pruned)", cv=5)
    plt.savefig('out/decision_tree_pruning/car-prune-learning.png')
    export_decision_tree(clf, 'car-prune')


def decision_tree_pruning_cancer():
    cancer_data = load_data_set('breastcancer')
    cancer_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    cancer_imp.fit(np.array(cancer_data['train']['inputs'] + cancer_data['test']['inputs'], dtype=np.float32))

    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="random"
    )

    with Timer() as t:
        clf.fit(cancer_imp.transform(cancer_data['train']['inputs']), cancer_data['train']['outputs'])

    time_to_fit = t.interval * 1000

    predicted = clf.predict(cancer_imp.transform(cancer_data['train']['inputs']))
    train_f1_score = metrics.f1_score(cancer_data['train']['outputs'], predicted, average='micro')

    with Timer() as t:
        predicted = clf.predict(cancer_imp.transform(cancer_data['test']['inputs']))
    test_f1_score = metrics.f1_score(cancer_data['test']['outputs'], predicted, average='micro')

    test_prediction_runtime = t.interval * 1000

    data_in = cancer_imp.transform(cancer_data['train']['inputs'] + cancer_data['test']['inputs'])
    data_out = cancer_data['train']['outputs'] + cancer_data['test']['outputs']

    t_out = cancer_data['test']['outputs']

    accuracy = accuracy_score(t_out, predicted) * 100
    precision = precision_score(t_out, predicted, average="weighted") * 100

    print("breastcancer.dataset (no pruning)")
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: Decision Trees (breastcancer.dataset, no pruning)", cv=5)
    plt.savefig('out/decision_tree_pruning/breastcancer-noprune-learning.png')
    export_decision_tree(clf, 'breastcancer-noprune')

    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
        min_samples_leaf=10,  # minimum of 10 samples at leaf nodes
        max_depth=5
    )

    with Timer() as t:
        clf.fit(cancer_imp.transform(cancer_data['train']['inputs']), cancer_data['train']['outputs'])

    time_to_fit = t.interval * 1000

    predicted = clf.predict(cancer_imp.transform(cancer_data['train']['inputs']))
    train_f1_score = metrics.f1_score(cancer_data['train']['outputs'], predicted, average='micro')

    with Timer() as t:
        predicted = clf.predict(cancer_imp.transform(cancer_data['test']['inputs']))
    test_f1_score = metrics.f1_score(cancer_data['test']['outputs'], predicted, average='micro')

    test_prediction_runtime = t.interval * 1000

    data_in = cancer_imp.transform(cancer_data['train']['inputs'] + cancer_data['test']['inputs'])
    data_out = cancer_data['train']['outputs'] + cancer_data['test']['outputs']

    t_out = cancer_data['test']['outputs']

    accuracy = accuracy_score(t_out, predicted) * 100
    precision = precision_score(t_out, predicted, average="weighted") * 100

    print("breastcancer.dataset (pruned)")
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: Decision Trees (breastcancer.dataset, pruned)", cv=5)
    plt.savefig('out/decision_tree_pruning/breastcancer-prune-learning.png')
    export_decision_tree(clf, 'breastcancer-prune')


def decision_tree_pruning(options):
    decision_tree_pruning_car()
    decision_tree_pruning_cancer()
    # plt.show()
    print("done")


def neural_net(options):
    print("neural_net")


def boosting(options):
    print("boosting")


def svm_car(kernel="linear"):
    car_data = load_data_set('car')
    car_ohe = preprocessing.OneHotEncoder()
    car_ohe.fit(car_data['train']['inputs'] + car_data['test']['inputs'])  # encode features as one-hot

    clf = tree.DecisionTreeClassifier(
        criterion="entropy",
        splitter="random",
    )

    with Timer() as t:
        clf.fit(car_ohe.transform(car_data['train']['inputs']), car_data['train']['outputs'])

    time_to_fit = t.interval * 1000

    predicted = clf.predict(car_ohe.transform(car_data['train']['inputs']))
    train_f1_score = metrics.f1_score(car_data['train']['outputs'], predicted, average='micro')

    with Timer() as t:
        predicted = clf.predict(car_ohe.transform(car_data['test']['inputs']))
    test_f1_score = metrics.f1_score(car_data['test']['outputs'], predicted, average='micro')

    test_prediction_runtime = t.interval * 1000

    data_in = car_ohe.transform(car_data['train']['inputs'] + car_data['test']['inputs'])
    data_out = car_data['train']['outputs'] + car_data['test']['outputs']

    t_out = car_data['test']['outputs']

    accuracy = accuracy_score(t_out, predicted) * 100
    precision = precision_score(t_out, predicted, average="weighted") * 100

    print("car.dataset (kernel={})".format(kernel))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: SVM (car.dataset, kernel={})".format(kernel), cv=5)
    plt.savefig('out/svm/car-kernel-{}.png'.format(kernel))


def svm_cancer(kernel="rbf"):
    cancer_data = load_data_set('breastcancer')
    cancer_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    cancer_imp.fit(np.array(cancer_data['train']['inputs'] + cancer_data['test']['inputs'], dtype=np.float32))

    clf = svm.SVC(
        kernel=kernel
    )

    with Timer() as t:
        clf.fit(cancer_imp.transform(cancer_data['train']['inputs']), cancer_data['train']['outputs'])

    time_to_fit = t.interval * 1000

    predicted = clf.predict(cancer_imp.transform(cancer_data['train']['inputs']))
    train_f1_score = metrics.f1_score(cancer_data['train']['outputs'], predicted, average='micro')

    with Timer() as t:
        predicted = clf.predict(cancer_imp.transform(cancer_data['test']['inputs']))
    test_f1_score = metrics.f1_score(cancer_data['test']['outputs'], predicted, average='micro')

    test_prediction_runtime = t.interval * 1000

    data_in = cancer_imp.transform(cancer_data['train']['inputs'] + cancer_data['test']['inputs'])
    data_out = cancer_data['train']['outputs'] + cancer_data['test']['outputs']

    t_out = cancer_data['test']['outputs']

    accuracy = accuracy_score(t_out, predicted) * 100
    precision = precision_score(t_out, predicted, average="weighted") * 100

    print("breastcancer.dataset (kernel={})".format(kernel))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: SVM (breastcancer.dataset, kernel={})".format(kernel), cv=5)
    plt.savefig('out/svm/breastcancer-kernel-{}.png'.format(kernel))



def _svm(options):
    kernel = options.kernel
    print(f"svm, kernel: {kernel}")
    svm_car(kernel=kernel)
    svm_cancer(kernel=kernel)
    print("done")


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
    parser_svm.set_defaults(func=_svm)
    parser_svm.add_argument(
        '-k', '--kernel', default='linear',
        help="Selects the kernel to use with SVM. Options: linear, rbf, poly. Defaults to linear.")

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
