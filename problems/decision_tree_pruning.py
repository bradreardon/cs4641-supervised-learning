# Analyze model complexity curve for NonBoost tree classifier, max_depth
'''file = open("coding_survey_nonboost_max_depth_results.csv", "w")
print("Beginning model complexity analysis for NonBoost...")
file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for max_depth in range(250):
    classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
    file.write(str(max_depth + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    file.write(str(classifier.score(input_list, output_list)) + ", ")
    file.write(str(classifier.score(test_input, test_output)) + "\n")'''

# Analyze model complexity curve for AdaBoost tree classifier, find n_estimators
'''file = open("coding_survey_adaboost_n_estimators_results.csv", "w")
print("Beginning model complexity analysis for AdaBoost... n_estimators")
file.write("n_estimators" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for n_estimators in range(50):
    classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=(n_estimators + 1) * 10)
    file.write(str((n_estimators + 1) * 10) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    file.write(str(classifier.score(input_list, output_list)) + ", ")
    file.write(str(classifier.score(test_input, test_output)) + "\n")'''

# Analyze model complexity curve for AdaBoost tree classifier, find max_depth
'''file = open("coding_survey_adaboost_max_depth_results.csv", "w")
print("Beginning model complexity analysis for AdaBoost... max_depth")
file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for max_depth in range(100):
    classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=50)
    result = ""
    result += (str(max_depth + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)'''

