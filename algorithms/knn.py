import numpy as np
import scikitplot as skplt
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics, impute
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from util import load_data_set, Timer


def knn_car(k_value=1):
    car_data = load_data_set('car')
    car_ohe = preprocessing.OneHotEncoder()
    car_ohe.fit(car_data['train']['inputs'] + car_data['test']['inputs'])  # encode features as one-hot

    x = list()
    y_train = list()
    y_test = list()
    y_cross = list()

    # chart different k-values vs. f1 score first
    for i in range(30):
        _k = i + 1
        clf = KNeighborsClassifier(n_neighbors=_k)
        clf.fit(car_ohe.transform(car_data['train']['inputs']), car_data['train']['outputs'])
        predicted = clf.predict(car_ohe.transform(car_data['train']['inputs']))
        train_f1_score = metrics.f1_score(car_data['train']['outputs'], predicted, average='micro')
        predicted = clf.predict(car_ohe.transform(car_data['test']['inputs']))
        test_f1_score = metrics.f1_score(car_data['test']['outputs'], predicted, average='micro')

        data_in = car_ohe.transform(car_data['train']['inputs'] + car_data['test']['inputs'])
        data_out = car_data['train']['outputs'] + car_data['test']['outputs']
        cross_val = cross_val_score(clf, data_in, data_out, cv=5)

        x.append(_k)
        y_train.append(train_f1_score)
        y_test.append(test_f1_score)
        y_cross.append(np.mean(cross_val))

    plt.figure()
    plt.title('Scores for various k (car.dataset)')
    plt.xlabel('k value')
    plt.ylabel('Score')
    plt.plot(x, y_train, label='Training F1 score')
    plt.plot(x, y_test, label='Testing F1 score')
    plt.plot(x, y_cross, label='Cross-validation score')
    plt.legend()
    plt.savefig('out/knn/car-k-testing.png')

    clf = KNeighborsClassifier(n_neighbors=k_value)

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

    print("car.dataset (k={})".format(k_value))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: kNN (car.dataset, k={})".format(k_value), cv=5)
    plt.savefig('out/knn/car-k-{}.png'.format(k_value))


def knn_cancer(k_value=1):
    cancer_data = load_data_set('breastcancer')
    cancer_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    cancer_imp.fit(np.array(cancer_data['train']['inputs'] + cancer_data['test']['inputs'], dtype=np.float32))

    x = list()
    y_train = list()
    y_test = list()
    y_cross = list()

    # chart different k-values vs. f1 score first
    for i in range(30):
        _k = i + 1
        clf = KNeighborsClassifier(n_neighbors=_k)
        clf.fit(cancer_imp.transform(cancer_data['train']['inputs']), cancer_data['train']['outputs'])
        predicted = clf.predict(cancer_imp.transform(cancer_data['train']['inputs']))
        train_f1_score = metrics.f1_score(cancer_data['train']['outputs'], predicted, average='micro')
        predicted = clf.predict(cancer_imp.transform(cancer_data['test']['inputs']))
        test_f1_score = metrics.f1_score(cancer_data['test']['outputs'], predicted, average='micro')

        data_in = cancer_imp.transform(cancer_data['train']['inputs'] + cancer_data['test']['inputs'])
        data_out = cancer_data['train']['outputs'] + cancer_data['test']['outputs']
        cross_val = cross_val_score(clf, data_in, data_out, cv=5)

        x.append(_k)
        y_train.append(train_f1_score)
        y_test.append(test_f1_score)
        y_cross.append(np.mean(cross_val))

    plt.figure()
    plt.title('Scores for various k (breastcancer.dataset)')
    plt.xlabel('k value')
    plt.ylabel('Score')
    plt.plot(x, y_train, label='Training F1 score')
    plt.plot(x, y_test, label='Testing F1 score')
    plt.plot(x, y_cross, label='Cross-validation score')
    plt.legend()
    plt.savefig('out/knn/breastcancer-k-testing.png')

    # chart with given k-value for detail
    clf = KNeighborsClassifier(n_neighbors=k_value)

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

    print("breastcancer.dataset (k={})".format(k_value))
    print("training f1 score:", train_f1_score)
    print("test f1 score:", test_f1_score)
    print("time to fit:", time_to_fit)
    print("test prediction runtime:", test_prediction_runtime)
    print("test accuracy", accuracy)
    print("test precision", precision)
    print()

    skplt.estimators.plot_learning_curve(
        clf, data_in, data_out, title="Learning Curve: kNN (breastcancer.dataset, k={})".format(k_value), cv=5)
    plt.savefig('out/knn/breastcancer-k-{}.png'.format(k_value))


def knn(options):
    # k = options.k_value
    # print(f"knn, k={k}")
    knn_car(k_value=11)
    knn_cancer(k_value=5)
    print("done")