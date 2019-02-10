import numpy as np
import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn import preprocessing, tree, metrics, impute
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, zero_one_loss
from sklearn.tree import DecisionTreeClassifier

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

    l_rate = .1
    save_error_graph(car_data, car_ohe, 'car-error-lrate-{}'.format(l_rate), {
                         "criterion": "gini",
                         "splitter": "random",
                         "min_samples_leaf": 5,
                         "max_depth": 9
                     },
                     max_n_estimators=250, learning_rate=l_rate, y_max=.4,
                     title="car.dataset, boosting (learning_rate={})".format(l_rate))


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

    l_rate = .1
    save_error_graph(cancer_data, cancer_imp, 'breastcancer-error-lrate-{}'.format(l_rate), {
                         "criterion": "gini",
                         "splitter": "random",
                         "min_samples_leaf": 10,
                         "max_depth": 5
                     },
                     max_n_estimators=1000, learning_rate=l_rate, y_max=.15,
                     title="breastcancer.dataset, boosting (learning_rate={})".format(l_rate))


def boosting(options):
    boosting_car(n_estimators=100)
    boosting_cancer(n_estimators=100)
    # plt.show()
    print("done")


def save_error_graph(dataset, transformer, figname, dt_params, max_n_estimators=1000, learning_rate=1., y_max=.4, title=None):
    X_train, y_train = transformer.transform(dataset['train']['inputs']), dataset['train']['outputs']
    X_test, y_test = transformer.transform(dataset['test']['inputs']), dataset['test']['outputs']

    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

    dt = DecisionTreeClassifier(**dt_params)
    dt.fit(X_train, y_train)
    dt_err = 1.0 - dt.score(X_test, y_test)

    ada_discrete = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=max_n_estimators,
        algorithm="SAMME")
    ada_discrete.fit(X_train, y_train)

    ada_real = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=max_n_estimators,
        algorithm="SAMME.R")
    ada_real.fit(X_train, y_train)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([1, max_n_estimators], [dt_stump_err] * 2, 'k-',
            label='Decision Stump Error')
    ax.plot([1, max_n_estimators], [dt_err] * 2, 'k--',
            label='Decision Tree Error')

    ada_discrete_err = np.zeros((max_n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
        ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

    ada_discrete_err_train = np.zeros((max_n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
        ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

    ada_real_err = np.zeros((max_n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
        ada_real_err[i] = zero_one_loss(y_pred, y_test)

    ada_real_err_train = np.zeros((max_n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
        ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

    ax.plot(np.arange(max_n_estimators) + 1, ada_discrete_err,
            label='Discrete AdaBoost Test Error',
            color='red')
    ax.plot(np.arange(max_n_estimators) + 1, ada_discrete_err_train,
            label='Discrete AdaBoost Train Error',
            color='blue')
    ax.plot(np.arange(max_n_estimators) + 1, ada_real_err,
            label='Real AdaBoost Test Error',
            color='orange')
    ax.plot(np.arange(max_n_estimators) + 1, ada_real_err_train,
            label='Real AdaBoost Train Error',
            color='green')

    ax.set_ylim((0.0, y_max))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')

    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)

    if title:
        plt.title(title)

    plt.savefig('out/boosting/{}.png'.format(figname))
