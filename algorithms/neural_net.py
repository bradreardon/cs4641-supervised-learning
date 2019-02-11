import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, impute, neural_network, preprocessing
from sklearn.metrics import accuracy_score, precision_score
import scikitplot as skplt

from util import Timer, load_data_set


def neural_net_cancer(solver):
    cancer_data = load_data_set('breastcancer')
    cancer_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    cancer_imp.fit(np.array(cancer_data['train']['inputs'] + cancer_data['test']['inputs'], dtype=np.float32))

    clf = neural_network.MLPClassifier(
        solver=solver, warm_start=True, max_iter=1000
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

    print("breastcancer.dataset (solver={})".format(solver))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: Neural Net (breastcancer.dataset, solver={})".format(solver), cv=5)
    plt.savefig('out/neural_net/breastcancer-solver-{}.png'.format(solver))


def neural_net_car(solver):
    car_data = load_data_set('car')
    car_ohe = preprocessing.OneHotEncoder()
    car_ohe.fit(car_data['train']['inputs'] + car_data['test']['inputs'])  # encode features as one-hot

    clf = neural_network.MLPClassifier(
        solver=solver, warm_start=True, max_iter=1000
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

    print("car.dataset (solver={})".format(solver))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: Neural Net (car.dataset, solver={})".format(solver), cv=5)
    plt.savefig('out/neural_net/car-solver-{}.png'.format(solver))


def neural_net(options):
    print("neural_net")
    for s in ['adam', 'sgd', 'lbfgs']:
        neural_net_cancer(solver=s)
        neural_net_car(solver=s)
