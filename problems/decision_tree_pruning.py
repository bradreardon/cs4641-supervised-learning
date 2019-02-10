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
            with open('car.data') as f:
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
