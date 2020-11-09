
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: cost
#######################################################################################################################
from utils import *
import numpy as np
import copy
from datetime import datetime

""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    begin = datetime.now()
    given_races = list(categorical_results.keys())
    
    total_dict = len(list(categorical_results.keys()))
    true_key_vals = list(categorical_results.values())
    
    key_vals = copy.deepcopy(true_key_vals)
    len_of_keyvals = [len(f) for f in key_vals]
    maximum_financial_loss = float("-inf")
    final_dictionary = {}
    
    dictionary_thresholds = {}
    dictionary = {given_races[0] : 0,
                  given_races[1] : 0,
                  given_races[2] : 0,
                  given_races[3] : 0
                  }
    a = np.linspace(0.0,1,101)
    cnt_no_change = 0
    break_all = False
    PPRs = []
    for i in range(total_dict):
        ppr_for_dict = []
        for j in a:
            thresh_holding = apply_threshold(key_vals[i],j)
            true_rate = get_num_predicted_positives(thresh_holding)
            true_rate /= len_of_keyvals[i]
            ppr_for_dict.append(true_rate)
        PPRs.append(ppr_for_dict)
        

    PPRs = np.round(PPRs, 4)

    trs_0 = PPRs[0]
    for j in range(100):
        trs_lst = trs_0[j]
        strt = trs_lst - epsilon
        end = trs_lst + epsilon
        tprs2 = np.array(PPRs[1])
        tprs3 = np.array(PPRs[2])
        tprs4 = np.array(PPRs[3])
        for i in range(100):
            lop = tprs2[i]
            lop_str = lop - epsilon
            lop_end = lop + epsilon
            if break_all:
                break
            elif (lop > strt and lop < end):
                for l in range(100):
                    lop1 = tprs3[l]
                    lop1_str = lop1 - epsilon
                    lop1_end = lop1 + epsilon
                    if break_all:
                        break
                    elif (lop1 > strt and lop1 < end) and (lop1 > lop_str and lop1 < lop_end):
                        for m in range(100):
              
                            lop2 = tprs4[m]

                            if break_all:
                                break
                            elif (lop2 > strt and lop2 < end) and (lop2 > lop1_str and lop2 < lop1_end) and (lop2 > lop_str and lop2 < lop_end):
                                trs = [(j)/100.,(i)/100.,(l)/100.,(m)/100.]
                                dictionary[given_races[0]] = apply_threshold(key_vals[0],trs[0])
                                dictionary[given_races[1]] = apply_threshold(key_vals[1],trs[1])
                                dictionary[given_races[2]] = apply_threshold(key_vals[2],trs[2])
                                dictionary[given_races[3]] = apply_threshold(key_vals[3],trs[3])
                                financial = apply_financials(dictionary)
                                if (financial) > (maximum_financial_loss):
                                    cnt_no_change = 0
                                    maximum_financial_loss = financial
                                    saved_trs = trs.copy()
                                    final_dictionary = {}
                                    final_dictionary = dictionary.copy()
                                elif float(financial) <= maximum_financial_loss and maximum_financial_loss != float('-inf'):
                                    if(cnt_no_change>1300):
                                        break_all = True
                                    cnt_no_change += 1
                                    break
                                
    for i in range(4):
        trs_hold = saved_trs[i]
        dictionary_thresholds[given_races[i]] = trs_hold
        
    end = datetime.now()

    seconds = end-begin
    print("Postprocessing took approximately: " + str(seconds) + " seconds")
        
    return final_dictionary, dictionary_thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    begin = datetime.now()
    given_races = list(categorical_results.keys())
 
    total_dict = len(list(categorical_results.keys()))
    true_key_vals = list(categorical_results.values())

    key_vals = copy.deepcopy(true_key_vals)
    maximum_financial_loss = float("-inf")
    final_dictionary = {}
    dictionary_thresholds = {}
    dictionary = {given_races[0] : 0,
                  given_races[1] : 0,
                  given_races[2] : 0,
                  given_races[3] : 0
                  }
    a = np.linspace(0.0,1,101)
    cnt_no_change = 0
    break_all = False
    PPRs = []
    for i in range(total_dict):
        ppr_for_dict = []
        for j in a:
            thresh_holding = apply_threshold(key_vals[i],j)
            true_rate = get_true_positive_rate(thresh_holding)
            ppr_for_dict.append(true_rate)
        PPRs.append(ppr_for_dict)
    PPRs = np.array(PPRs, dtype='float')
    PPRs = np.round(PPRs, 4)

    trs_0 = PPRs[0]
    for j in range(100):
        trs_lst = trs_0[j]
        strt = trs_lst - epsilon
        end = trs_lst + epsilon
        tprs2 = np.array(PPRs[1])
        tprs3 = np.array(PPRs[2])
        tprs4 = np.array(PPRs[3])
        for i in range(100):
            lop = tprs2[i]
            lop_str = lop - epsilon
            lop_end = lop + epsilon
            if break_all:
                break
            elif (lop > strt and lop < end):
                for l in range(100):
                    lop1 = tprs3[l]
                    lop1_str = lop1 - epsilon
                    lop1_end = lop1 + epsilon
                    if break_all:
                        break
                    elif (lop1 > strt and lop1 < end) and (lop1 > lop_str and lop1 < lop_end):
                        for m in range(100):
                            #print(m)
                            lop2 = tprs4[m]

                            if break_all:
                                break
                            elif (lop2 > strt and lop2 < end) and (lop2 > lop1_str and lop2 < lop1_end) and (lop2 > lop_str and lop2 < lop_end):
                                trs = [(j)/100.,(i)/100.,(l)/100.,(m)/100.]
                                dictionary[given_races[0]] = apply_threshold(key_vals[0],trs[0])
                                dictionary[given_races[1]] = apply_threshold(key_vals[1],trs[1])
                                dictionary[given_races[2]] = apply_threshold(key_vals[2],trs[2])
                                dictionary[given_races[3]] = apply_threshold(key_vals[3],trs[3])
                                financial = apply_financials(dictionary)
                                if (financial) > (maximum_financial_loss):
                                    cnt_no_change = 0
                                    maximum_financial_loss = financial
                                    saved_trs = trs.copy()
                                    final_dictionary = {}
                                    final_dictionary = dictionary.copy()
                                elif float(financial) <= maximum_financial_loss and maximum_financial_loss != float('-inf'):
                                    if(cnt_no_change>1500):
                                        break_all = True
                                    cnt_no_change += 1
                                    break
                            
                            
    for i in range(4):
        trs_hold = saved_trs[i]
        dictionary_thresholds[given_races[i]] = trs_hold
          
    end = datetime.now()

    seconds = end-begin
    print("Postprocessing took approximately: " + str(seconds) + " seconds")

    return final_dictionary, dictionary_thresholds


#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    begin = datetime.now()
    given_races = list(categorical_results.keys())
    true_key_vals = list(categorical_results.values())

    key_vals = copy.deepcopy(true_key_vals)
    #maximum_financial_loss = float("-inf")

    final_dictionary = {}

    dictionary_thresholds = {}
    a = np.linspace(0.0,1,101)
    
    maximum_financial_loss = [float("-inf"),float("-inf"),float("-inf"),float("-inf")]
    for i in a:
        for m in range(4):
            temp_dict = []
            temp_dict = apply_threshold(key_vals[m],i)
            financial = apply_financials(temp_dict, True)
            if financial>maximum_financial_loss[m]:
                maximum_financial_loss[m] = financial
                dictionary_thresholds[given_races[m]] = i
                final_dictionary[given_races[m]] = temp_dict
 
    end = datetime.now()

    seconds = end-begin
    print("Postprocessing took approximately: " + str(seconds) + " seconds")

    return final_dictionary, dictionary_thresholds
   

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    begin = datetime.now()
    given_races = list(categorical_results.keys())
    total_dict = len(list(categorical_results.keys()))
    true_key_vals = list(categorical_results.values())
    key_vals = copy.deepcopy(true_key_vals)
    maximum_financial_loss = float("-inf")
    final_dictionary = {}
    dictionary_thresholds = {}
#     dictionary = {given_races[0] : 0,
#                   given_races[1] : 0,
#                   given_races[2] : 0,
#                   given_races[3] : 0
#                   }
    dictionary = {}
    TPRS = []
    break_all = False
    cnt_no_change = 0
    a = np.linspace(0.0,1,101)
    for i in range(total_dict):
        tpr_for_dict = []
        for j in a:
            thresh_holding = apply_threshold(key_vals[i],j)
            true_rate = get_positive_predictive_value(thresh_holding)
            tpr_for_dict.append(true_rate)
        TPRS.append(tpr_for_dict)
    trs_0 = TPRS[0]
    PPRs = np.array(TPRS, dtype='float')
    PPRs = np.round(PPRs, 4)
    trs_0 = PPRs[0]
    for j in range(100):
        trs_lst = trs_0[j]
        strt = trs_lst - epsilon
        end = trs_lst + epsilon
        tprs2 = np.array(PPRs[1])
        tprs3 = np.array(PPRs[2])
        tprs4 = np.array(PPRs[3])
        for i in range(100):
            lop = tprs2[i]
            lop_str = lop - epsilon
            lop_end = lop + epsilon
            if break_all:
                break
            elif (lop > strt and lop < end):
                for l in range(100):
                    lop1 = tprs3[l]
                    lop1_str = lop1 - epsilon
                    lop1_end = lop1 + epsilon
                    if break_all:
                        break
                    elif (lop1 > strt and lop1 < end) and (lop1 > lop_str and lop1 < lop_end):
                        for m in range(100):
                            #print(m)
                            lop2 = tprs4[m]
                            if break_all:
                                break
                            elif (lop2 > strt and lop2 < end) and (lop2 > lop1_str and lop2 < lop1_end) and (lop2 > lop_str and lop2 < lop_end):
                                trs = [(j)/100.,(i)/100.,(l)/100.,(m)/100.]
                                dictionary[given_races[0]] = apply_threshold(key_vals[0],trs[0])
                                dictionary[given_races[1]] = apply_threshold(key_vals[1],trs[1])
                                dictionary[given_races[2]] = apply_threshold(key_vals[2],trs[2])
                                dictionary[given_races[3]] = apply_threshold(key_vals[3],trs[3])
                                financial = apply_financials(dictionary)
                                if (financial) > (maximum_financial_loss):
                                    cnt_no_change = 0
                                    maximum_financial_loss = financial
                                    saved_trs = trs.copy()
                                    final_dictionary = {}
                                    final_dictionary = dictionary.copy()
                                elif float(financial) <= maximum_financial_loss and maximum_financial_loss != float('-inf'):
                                    if(cnt_no_change>2000):
                                        break_all = True
                                        pass
                                    cnt_no_change += 1
                                    break
                        
        
    for i in range(4):
        trs_hold = saved_trs[i]
        dictionary_thresholds[given_races[i]] = trs_hold
    end = datetime.now()

    seconds = end-begin
    print("Postprocessing took approximately: " + str(seconds) + " seconds")
    return final_dictionary, dictionary_thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
    categorical_results : Dictionary with each group value as keys. Each key has a list of (prediction, label) tuples
representing all of the data points within that group
"""

def enforce_single_threshold(categorical_results):
    begin = datetime.now()
    given_races = list(categorical_results.keys())
    true_key_vals = list(categorical_results.values())
    key_vals = copy.deepcopy(true_key_vals)
    maximum_financial_loss = float("-inf")
    final_dictionary = {}
    dictionary_thresholds = {}
    final_trs = 0.0
    a = np.linspace(0.0,1,101)
    for i in a:
        dictionary = {}
        for j in range(4):
            thresh_holding_1 = apply_threshold(key_vals[j] , i)
            dictionary[given_races[j]] = thresh_holding_1
        financial = apply_financials(dictionary)
        if financial > maximum_financial_loss:
            maximum_financial_loss = financial
            final_dictionary = {}
            final_dictionary = dictionary.copy()
            final_trs = i
        
    for i in range(4):
        dictionary_thresholds[given_races[i]] = final_trs

    end = datetime.now()

    seconds = end-begin
    print("Postprocessing took approximately: " + str(seconds) + " seconds")        
    return final_dictionary, dictionary_thresholds

