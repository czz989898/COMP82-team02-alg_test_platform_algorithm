import os
import glob
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import defaultdict as dd
from metrics import metrics
from langid.langid import LanguageIdentifier, model
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
#this class is for write json
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
def test():
    user_dict = calc_stats(metrics, nmin=4, nmax=4, max_feats=1000,
                           ngram_type="char_wb")
    print(user_dict['probability'])
    print(user_dict['threshold'])
    with open('./probability.json', 'w') as jsonfile:
        json.dump(user_dict['probability'], jsonfile, cls=NpEncoder)
    with open('./threshold.json', 'w') as jsonfile:
        json.dump(user_dict['threshold'], jsonfile, cls=NpEncoder)
    # ----------------------------------draw graph-----------------------------
    label_list = list(user_dict['probability'].keys())
    x = range(len(list(user_dict['probability'].values())))
    plt.figure(figsize=(10, 6), dpi=100)
    plt.bar(x, list(user_dict['probability'].values()), color='blue')
    plt.ylim(0, 1)
    plt.xticks(x, label_list)
    plt.ylabel("probability")
    plt.xlabel("matrixName")
    for a, b in zip(x, list(user_dict['probability'].values())):
        plt.text(a, b + 0.05 ,'%.3f' % b, ha='center', va= 'bottom',fontsize=7)
    plt.show()
    #plt.savefig('probability.png')
def calc_stats(metrics, nmin, nmax, max_feats, ngram_type):
    '''
    metrics (dict): Mapping from metric names to the functions.
                    E.g. {"cosine": cosine_similarity, ...}
    nmin
    nmax
    max_feats
    ngram_type
    '''

    # Create a dictionary containing users and their histories and new docs.
    user_dict = get_content_from_docs()
    # Access user's history (list of strings) & new file ([str] of len 1)
    contents = user_dict["known"]
    unknown_content = user_dict["unknown"]
    word_matrix, unknown_doc_vector = get_ngrams(contents, unknown_content,
                                                     nmin, nmax, max_feats,
                                                     ngram_type)

    # Calc & store metrics (e.g. cosine distance)
    user_dict['probability']={}
    for metric_name, metric in metrics.items():
        user_dict['probability'][metric_name] = metric(word_matrix,unknown_doc_vector)
    #-----------------------------------------------------------------------------------
    user_dict['threshold'] = {}
    for metric_name, metric in metrics.items():
        user_dict['threshold'][metric_name] = 0
        for i in range(0,len(user_dict["known"])):
            new_contents=[]
            for j in range(0,len(user_dict["known"])):
                if j !=i:new_contents.append(user_dict["known"][j])
            word_matrix1, word_matrix2 = get_ngrams(new_contents, [user_dict["known"][i]],
                                                         nmin, nmax, max_feats,
                                                         ngram_type)
            user_dict['threshold'][metric_name] += metric(word_matrix1, word_matrix2)
        user_dict['threshold'][metric_name]=user_dict['threshold'][metric_name]/len(user_dict["known"])
    # Detect language of each new unknown file
    # for user in tqdm(user_dict, ascii=True, desc="Detecting languages"):
    #     user_dict[user]["language"] = get_lang(
    #         ' '.join(user_dict[user]["unknownfile"]))

    return user_dict
def get_lang(text_string):
    return LanguageIdentifier.from_modelstring(model,
               norm_probs=True).classify(text_string)[0]


def txt_to_str(fname):
    ###########################################这里我加上了utf-8！！！！！！！！！！！！！！！！###########################
    with open(fname,encoding = 'utf-8') as f:
        return "".join([x.strip() for x in f.readlines()])


def get_content_from_docs():
    '''
    Return a mapping of user (folder) names to each user's history, new files,
    etc.
    '''
    user_dict = dd(dict)
    user_dict['known']=[]
    user_dict['unknown'] = []
    for user_folder in glob.glob(r"./data/*"):

        if '.txt' in user_folder:
            continue
        # Initialise user dict entry
        user = user_folder.split("/")[-1].split('\\')[-1]
        # Add to user's history and the unknown file
        if user=='known':
            for file in glob.glob(user_folder+'/*'):
                if 'Timer' not in file:
                    user_dict[user].append(txt_to_str(file))
        if user == 'unknown':
            for unknownfile in glob.glob(user_folder+'/*'):
                if 'Timer' not in unknownfile:
                    user_dict[user].append(txt_to_str(unknownfile))
    return user_dict
def del_file(path_data):
    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)



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
if __name__ == "__main__":
    #todo same_test()
    #todo diff_test()
    test()
    del_file('./data')
    #get_content_from_docs()