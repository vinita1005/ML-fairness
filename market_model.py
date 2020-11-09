from sklearn import svm
from Preprocessing import preprocess
from Postprocessing import *
from utils import *
import numpy as np

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=10000)
SVR.fit(training_data, training_labels)

training_class_predictions = SVR.predict(training_data)
training_predictions = []
test_class_predictions = SVR.predict(test_data)
test_predictions = []


for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i])



training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)



########enforce equal oppurtunity#########
print("enforcing equal oppurtunity")
training_race_cases_equal_op, thresholds_ep = enforce_equal_opportunity(training_race_cases,0.01)
test_race_cases_equal_op = test_race_cases.copy()

for group in test_race_cases_equal_op.keys():
    test_race_cases_equal_op[group] = apply_threshold(test_race_cases_equal_op[group], thresholds_ep[group])
    
print("For equal oppurtunity the true positive rates are within mentioned epilon i.e 0.01")

print("")
for group in training_race_cases_equal_op.keys():
    TPR = get_true_positive_rate(training_race_cases_equal_op[group])
    print("TPR for " + group + ": " + str(TPR))
    
print("")
for group in training_race_cases_equal_op.keys():
    FPR = get_false_positive_rate(training_race_cases_equal_op[group])
    print("FPR for " + group + ": " + str(FPR))
    
print("")    
for group in training_race_cases_equal_op.keys():
    accuracy = get_num_correct(training_race_cases_equal_op[group]) / len(training_race_cases_equal_op[group])
    print("Accuracy in Training data for " + group + ": " + str(accuracy))

print("")    
for group in test_race_cases_equal_op.keys():
    accuracy = get_num_correct(test_race_cases_equal_op[group]) / len(test_race_cases_equal_op[group])
    print("Accuracy in testing data for " + group + ": " + str(accuracy))


print("")
print("Accuracy on Training data:")
print(get_total_accuracy(training_race_cases_equal_op))
print("")

print("Cost on Training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases_equal_op)))
print("")
   

print("")
print("Accuracy on test data:")
print(get_total_accuracy(test_race_cases_equal_op))
print("")

print("Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases_equal_op)))
print("")
   


# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED
