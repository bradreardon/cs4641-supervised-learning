import numpy as np
import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn import preprocessing, tree, metrics, impute
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score

from util import load_data_set, Timer


def boosting_car(n_estimators=1):
    car_data = load_data_set('car')
    car_ohe = preprocessing.OneHotEncoder()
    car_ohe.fit(car_data['train']['inputs'] + car_data['test']['inputs'])  # encode features as one-hot

    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
        min_samples_leaf=5,  # minimum of 5 samples at leaf nodes
        max_depth=9
    ), n_estimators=n_estimators, algorithm="SAMME")

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

    print("car.dataset (n_estimators={})".format(n_estimators))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()
    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out,
        title="Learning Curve: Boosting (car.dataset, n_estimators={})".format(n_estimators), cv=5)
    plt.savefig('out/boosting/car-estimators-{}.png'.format(n_estimators))


def boosting_cancer(n_estimators=1):
    cancer_data = load_data_set('breastcancer')
    cancer_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    cancer_imp.fit(np.array(cancer_data['train']['inputs'] + cancer_data['test']['inputs'], dtype=np.float32))

    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
        min_samples_leaf=10,  # minimum of 10 samples at leaf nodes
        max_depth=5
    ), n_estimators=n_estimators, algorithm="SAMME.R")

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

    print("breastcancer.dataset (n_estimators={})".format(n_estimators))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out,
        title="Learning Curve: Boosting (breastcancer.dataset, n_estimators={})".format(n_estimators), cv=5)
    plt.savefig('out/boosting/breastcancer-estimators-{}.png'.format(n_estimators))


def boosting(options):
    boosting_car(n_estimators=100)
    boosting_cancer(n_estimators=100)
    # plt.show()
    print("done")