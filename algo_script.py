'''
Name: app/algo/algo_script.py
Author: David Stern - Team IC.
Purpose: Tests metrics on PAN13 examples, calculating thresholds and accuracy
achieved given those thresholds.

Use: With the virtual env active, run `python3.7 -m app.algo.algo_script` to
'''

import os
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import defaultdict as dd
from metrics import metrics
from langid.langid import LanguageIdentifier, model
from tqdm import tqdm
import json
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numpy import trapz

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
def same_test(unknown):
    unknown = '0'+str(unknown)
    # Get the similarity scores for each user
    user_dict = calc_stats(metrics, nmin=4, nmax=4, max_feats=2000,
                           ngram_type="char_wb",unknown=unknown)
    if user_dict==False:return False
    truth_data = pd.read_csv("./data/truth.txt", sep=" ", header=None,
                             # We will name the value column 'val'
                             names=['val'],
                             # The first col for each row = user/folder names
                             index_col=0,
                             # Convert column 1 to 1 if "Y" else 0.
                             converters={1: lambda x: 1 if x == "Y" else 0})
    # Get the truth data & add it to the user_dict
    # Note: This file must be run from the main repo folder via the command:
    #           `python -m app.algo.algo_script`

    # Select & store truth values from truth_data

    for user in user_dict:
        print(truth_data['val'].get(int(user[-2:])))
        user_dict[user]["truth"] = truth_data['val'].get(int(user[-2:]))
        # ^By row, ^then by col
    # Very preliminary analysis of the metrics
    labeled_results = {}  # Useful for accesing by user, can be used later
    tfs = {name: ([], []) for name in metrics.keys()}
    thresholds = {name: -1 for name in metrics.keys()}
    for user in user_dict:
        labeled_results[user] = {}
        labeled_results[user]['truth'] = user_dict[user]['truth']
        for metric_name in metrics:
            labeled_results[user][metric_name] = user_dict[user][metric_name]
            # print(labeled_results[user][metric_name])
        tf_list = 0 if user_dict[user]['truth'] else 1
        for metric_name in metrics.keys():
            tfs[metric_name][tf_list].append(labeled_results[user][metric_name])

    # add this line to print label_result
    # Print each group's mean, min and max
    print(f"{65 * '-'}\n| Metric\t\t| Class\t\t| Min\t| Mean\t| Max\t|\n{65 * '-'}")
    for metric_name, (trues, falses) in tfs.items():
        # Add this line to print
        # print(metric_name, (trues, falses))
        #for (vals, name) in ((trues, "Trues"), (falses, "Falses")):
        name="Trues"
        print(f"| {metric_name:<10}\t| {name:<6}:\t| {np.amin(trues):.2}\t",
                  f"| {np.mean(trues):.2}\t| {np.amax(trues):.2}\t|")

        # Threshold Calculation - simply the mean value of the metric.
        thresholds[metric_name] = (np.mean(trues + falses))
    print(65 * '-')

    # add this line to print thresholds
    # print(thresholds)
    # Threshold result
    # This threshold is used to judge whether a method judges a document as
    # written by the author of the set of documents in the history, or not.
    # We do not take responsibility for this method or its accuracy.
    print("\nThreshold Calculations")
    for metric_name, threshold in thresholds.items():
        trues, falses = tfs[metric_name]
        total = len(trues + falses)
        correct_trues = (trues >= threshold).sum()
        correct_falses = (falses < threshold).sum()
        incorrect_trues = (trues < threshold).sum()
        incorrect_falses = (falses >= threshold).sum()
        accuracy = (correct_trues + correct_falses) / len(trues + falses)
        print(f"{metric_name:<10}:\t With threshold={threshold:.4},",
              f"accuracy={accuracy:.4}")
    print('\nAnalysis Testing....')
    result_analysis(thresholds, labeled_results,unknown)


def calc_stats(metrics, nmin, nmax, max_feats, ngram_type,unknown):
    '''
    metrics (dict): Mapping from metric names to the functions.
                    E.g. {"cosine": cosine_similarity, ...}
    nmin
    nmax
    max_feats
    ngram_type
    '''

    # Create a dictionary containing users and their histories and new docs.
    user_dict = get_content_from_docs(unknown)
    if user_dict==False:return False
    print(list(user_dict.keys()))
    for user in tqdm(user_dict, ascii=True, desc="Calculating metrics"):

        # Access user's history (list of strings) & new file ([str] of len 1)
        contents = user_dict[user]["history"]
        unknown_content = user_dict[user]["unknownfile"]
        word_matrix, unknown_doc_vector = get_ngrams(contents, unknown_content,
                                                     nmin, nmax, max_feats,
                                                     ngram_type)

        # Calc & store metrics (e.g. cosine distance)
        for metric_name, metric in metrics.items():
            user_dict[user][metric_name] = metric(word_matrix,
                                                  unknown_doc_vector)

    # Detect language of each new unknown file
    # for user in tqdm(user_dict, ascii=True, desc="Detecting languages"):
    #     user_dict[user]["language"] = get_lang(
    #         ' '.join(user_dict[user]["unknownfile"]))

    return user_dict


def get_lang(text_string):
    return LanguageIdentifier.from_modelstring(model,
                                               norm_probs=True).classify(text_string)[0]


def txt_to_str(fname):
    with open(fname) as f:
        return "".join([x.strip() for x in f.readlines()])


def get_content_from_docs(unknown):
    '''
    Return a mapping of user (folder) names to each user's history, new files,
    etc.
    '''
    user_dict = dd(dict)
    truth = {}
    with open('./data/truth.txt','w')as truth_file:
        for user_folder in glob.glob(r"./data/*"):
            print(user_folder)

            if '.txt' in user_folder:
                continue

            # Initialise user dict entry
            user = user_folder.split("/")[-1]
            user_dict[user]["history"] = []
            username=user.split('\\')[-1]
            truth_file.write(username+" Y\n")
            for data_folder in glob.glob(r"./"+user+'/*'):

                # Add to user's history and the unknown file
                data = data_folder.split("/")[-1]

                if 'log' not in data_folder:
                    for file in glob.glob(os.path.join(data_folder, '*.txt')):
                        if unknown in file and 'Time' in file:return False
                        if 'Timer' not in file:
                            if unknown not in file:
                                user_dict[user]["history"].append(txt_to_str(file))
                            else:
                                user_dict[user]["unknownfile"] = [txt_to_str(file)]
    return user_dict


def get_ngrams(history, new_doc, min_n, max_n, max_feats, analyzer):
    '''
    history (list(str)): A list with strings representing past documents whose
                         authorship is known.
    new_doc (list(str)): A list with a single string representing the document
                         content being analysed.
    min_n:
    max_n:
    max_feats:
    analyzer:
    '''
    # Create a vector of [doc1: {word1_count, word2_count}, doc2: {...}, ...]
    vectoriser = CountVectorizer(ngram_range=(min_n, max_n), analyzer=analyzer,
                                 max_features=max_feats)

    # Get the top n-grams & get the n-gram count in the new document
    result = vectoriser.fit_transform(history).toarray()
    # word_matrix = np.sum(result, axis=0)
    word_matrix = result
    debug = False
    if debug:
        print(f"Result (dimensions, shape): ({result.ndim}, {result.shape})")
        print(f"word_matrix (dimns, shape): ({word_matrix.ndim}, {word_matrix.shape})")
        print("result: ", result)
        print("word_matrix: ", word_matrix)
    # word_matrix = np.sum(vectoriser.fit_transform(history).toarray(),
    #                      axis=0)
    unknown_vector = vectoriser.transform(new_doc).toarray()

    # # Reshape the data using X.reshape(1, -1); treat it as a single sample
    # word_matrix = word_matrix.reshape(1, -1)
    if debug:
        print("\nAfter reshape(1, -1):")
        print(f"word_matrix (dimns, shape): ({word_matrix.ndim}, {word_matrix.shape})")
        print("word_matrix: ", word_matrix)
    unknown_vector = unknown_vector.reshape(1, -1)

    if debug:
        error = 1 + "e"

    # Return the ngram counts of the history and the new document to be assessed
    return word_matrix, unknown_vector


def result_analysis(thresholds, labeled_results,unknown):
    result_list = []
    for metric_name, threshold in thresholds.items():
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for user, data in labeled_results.items():
            name = user.split('\\')[-1]
            truth = data['truth']
            if data[metric_name] >= threshold:
                prediction = 1
            else:
                prediction = 0
            if truth == 1 and prediction == 1:
                true_positive = true_positive + 1
                type = 'True_Positive'
            elif truth == 1 and prediction == 0:
                false_negative = false_negative + 1
                type = 'False_Negative'
            elif truth == 0 and prediction == 1:
                false_positive = false_positive + 1
                type = 'False_Positive'
            else:
                true_negative = true_negative + 1
                type = 'True_Negative'
            r_dict = {'matrixName':metric_name,'File': name, 'Truth': truth, 'Prediction': prediction, 'result': type,'question':unknown}
            result_list.append(r_dict)

        #accuracy = (true_positive + true_negative) / len(labeled_results)
        #precision = true_positive / (true_positive + false_positive)
        #recall = true_positive / (true_positive + false_negative)
        #F1_score = (2 * precision * recall) / (precision + recall)
        #print('metric_name:',metric_name)
        #print('resultTF:',result_list)
        with open('./tf_results_temp/'+unknown+'.json','w') as jsonfile:
            json.dump(result_list,jsonfile,cls=NpEncoder)
        #print('metric name:',metric_name)
        #print('accuracy:',accuracy)
        #print('precision',precision)
        #print('recall',recall)
        #print('F1_score',F1_score)

'''
def draw_auc(metric_name, labeled_results):
    accuracy_list = []
    Lk_list = []
    thresholds = []
    for threshold in np.arange(0.0, 1.01, 0.05):
        thresholds.append(threshold)
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for user, data in labeled_results.items():
            name = user
            truth = data['truth']
            if data[metric_name] >= threshold:
                prediction = 1
            else:
                prediction = 0
            if truth == 1 and prediction == 1:
                true_positive = true_positive + 1
                type = 'True_Positive'
            elif truth == 1 and prediction == 0:
                false_negative = false_negative + 1
                type = 'False_Negative'
            elif truth == 0 and prediction == 1:
                false_positive = false_positive + 1
                type = 'False_Positive'
            else:
                true_negative = true_negative + 1
                type = 'True_Negative'
        print(true_positive,false_negative,false_positive,true_negative)
        accuracy = (true_positive+true_negative)/(false_negative+false_positive)
        accuracy_list.append(accuracy)
        Lk_list.append(threshold)

    # print('\nThe threshold list is \n')
    # print(thresholds)

    # tpr.append(0.0)
    # fpr.append(0.0)
    # print(tpr)
    # print(fpr)
    # print(sorted(tpr))
    # print(sorted(fpr))
    plt.figure()
    plt.plot(accuracy_list, Lk_list, linewidth='1', color='blue')
    # plt.plot(sorted(fpr),sorted(tpr))
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    my_x_ticks = np.arange(0, 1.1, 0.1)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.title("ROC")
    # plt.show()
    plt.savefig("./ROC.jpg")

    print('AUC is: \n')
    print(trapz(sorted(tpr), sorted(fpr), dx=0.001))

'''
def create_results():
    resultTF=[]
    f1table=[]
    metric_info={}#{'metric_name':[true_positive,false_positive,true_negative,false_negative]}
    for json_file in glob.glob(r"./tf_results_temp/*"):
        with open(json_file,'r') as data_file:
            data=json.load(data_file)
            for dic in data:
                resultTF.append(dic)
                if dic['matrixName'] not in metric_info:
                    metric_info[dic['matrixName']]=[0,0,0,0]
                else:
                    if dic['result']=='True_Positive':metric_info[dic['matrixName']][0]+=1
                    elif dic['result']=='True_Negative':metric_info[dic['matrixName']][1]+=1
                    elif dic['result'] == 'False_Positive':metric_info[dic['matrixName']][2] += 1
                    elif dic['result'] == 'False_Negative': metric_info[dic['matrixName']][3] += 1
    with open('./resultTF.json', 'w') as jsonfile:
        json.dump(resultTF, jsonfile, cls=NpEncoder)
    print(metric_info)
    f1_list=[]
    for k,v in metric_info.items():
        # precision = true_positive / (true_positive + false_positive)
        # recall = true_positive / (true_positive + false_negative)
        # F1_score = (2 * precision * recall) / (precision + recall)
        precision = v[0]/(v[0]+v[2])
        recall = v[0]/(v[0]+v[3])
        f1score = (2 * precision * recall) / (precision + recall)
        f1table.append({'matrixName':k,'f1':f1score})
        f1_list.append(f1score)
    with open('./f1table.json', 'w') as jsonfile:
        json.dump(f1table, jsonfile, cls=NpEncoder)
    #----------------------------------draw graph-----------------------------
    label_list = list(metric_info.keys())
    x = range(len(f1_list))
    plt.figure(figsize=(10, 6), dpi=100)
    plt.bar(x, f1_list,color='blue')
    plt.ylim(0, 1)
    plt.xticks(x,label_list)
    plt.ylabel("f1_score")
    plt.xlabel("matrixName")
    plt.savefig('f1_table.png')
if __name__ == "__main__":
    #todo same_test()
    #todo diff_test()
    for i in range(2,10):
        print(i)
        same_test(i)
    create_results()
