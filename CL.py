import csv
import math
import os
from time import time

import sklearn
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from tqdm import tqdm


def exponential(i):
    """
    This function is the pacing function-Varied exponential pacing
    :param i: The batch number
    :type i: integer
    :return: The batch size
    :rtype: integer
    """
    return int(math.pow(2,(i)))

def svm_score(data,class_index=-1):
    """
    This function is the scoring function
    :param data: The data
    :type data: ndarray
    :param class_index: The target index
    :type class_index: int
    :return: The data indexes are sorted by the rank
    :rtype:
    """
    y_train= data[:, class_index:]
    x_train= data[:, :class_index]
    train_score=transfer_values_svm_scores(x_train,y_train)
    return rank_data_according_to_score(train_score,y_train)

def get_data_by_rank(data,rank):
    """
    This function returns the data sorted according to the ranking
    :param data: The data
    :type data:ndarray
    :param rank:The rank
    :type rank:list
    :return:The data sorted according to the ranking
    :rtype:ndarray
    """
    new_data=[]
    data=(np.array(data))
    for i in rank:
        record=data[i]
        new_data.append(record)
    return np.array(new_data)

def CL(data,pacing_func=exponential,score_func=svm_score):
    """
    This function is the implementation of the CL algorithm from the paper
    :param data:The dataset
    :type data:ndarray
    :param pacing_func:The pacing function
    :type pacing_func:function
    :param score_func:The score function
    :type score_func:function
    :return:The data batches
    :rtype:list of ndarray
    """
    rank=score_func(data)
    new_data=get_data_by_rank(data,rank)
    result=[]
    rounds=int(np.ceil(math.log(data.shape[0],2)))
    for i in range(1,rounds):
        size=min(pacing_func(i+1),data.shape[0])
        new_x=new_data[:size]
        result.append(new_x)
    return result

def transfer_values_svm_scores(train_x, train_y):
    """
    This function calculates the score
    :param train_x: The train data features
    :type train_x: ndarray
    :param train_y:The train data label
    :type train_y:ndarray
    :return:The scores for the train data
    :rtype:ndarray
    """
    clf = svm.SVC(probability=True)
    print("fitting svm")
    clf.fit(train_x, train_y)
    train_scores = clf.predict_proba(train_x)
    return train_scores

def rank_data_according_to_score(train_scores, y_train, reverse=False, random=False):
    """
    This function returns the indexes of the records sorted by the score
    :param train_scores: The train data score
    :type train_scores: ndarray
    :param y_train: The train label
    :type y_train: ndarray
    :return: The indexes of the records sorted by the score
    :rtype: ndarray
    """
    train_size, _ = train_scores.shape
    y_train=np.array(y_train)
    train_scores=np.array(train_scores)
    y_train=y_train.reshape(-1)
    y_train=y_train.astype(int)
    hardness_score = train_scores[list(range(train_size)), y_train]
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    if reverse:
        res = np.flip(res, 0)
    if random:
        np.random.shuffle(res)
    return res


def write(datasets_name, algo_name,data,test_index,best_clf,t_time,best_params,cv):
    """
    This function writes the results to the results file
    :param datasets_name: The dataset name
    :type datasets_name:String
    :param algo_name:Tha algorithm name
    :type algo_name: String
    :param data: The dataset
    :type data:ndarray
    :param test_index:The test records index
    :type test_index:list
    :param best_clf:The best classifier
    :type best_clf:classifier
    :param t_time:Training time
    :type t_time:Float
    :param best_params:The bast classifier params
    :type best_params: dictionary
    :param cv: The cross validation iteration
    :type cv:Integer
    """
    test_data = data[test_index]
    y_test = np.array(test_data[:, -1:])
    x_test = test_data[:, :-1]

    pred_time = time()

    pred = best_clf.predict(x_test)

    pred_time = time() - pred_time
    pred_time = pred_time / y_test.shape[0]
    pred_time = pred_time * 1000

    acc = accuracy_score(y_test, pred)

    y_score = best_clf.predict_proba(x_test)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, pred)
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = precision_score(y_test, pred)

    tpr = tp / (tp + fn)
    fpr = 1 - tpr

    auc = roc_auc_score(y_test, y_score[:, 1])

    pr_curve = average_precision_score(y_test, y_score[:, 1])

    with open('./results.csv', 'a') as f:
        w = csv.writer(f)
        row = [datasets_name, algo_name, str(cv), best_params, str(acc), str(tpr), str(fpr), str(precision), str(auc),str(pr_curve), str(t_time), str(pred_time)]
        w.writerow(row)


def main(path):
    """
    The main function
    :param path: Path to the DS
    :type path: String
    """
    algo_name="CL"
    df=pd.read_csv(path)
    datasets_name=path.split('/')[-1]
    data=np.array(df)
    cv_outer = KFold(n_splits=10, shuffle=True)
    random_grid= {"n_estimators":[10,20,30,40],
                  "max_depth":list(range(10,50)),
                  "criterion":["gini","entropy"]
    }
    cv=0
    for train_index, test_index in tqdm(cv_outer.split(data)):
        cv+=1
        batches=CL(data[train_index])
        clf = RandomForestClassifier()
        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0, random_state=42)
        t_time=time()
        for batch in batches:
            try:
                data_batch=np.array(batch)
                y_batch=data_batch[:, -1:]
                x_batch= data_batch[:, :-1]
                clf.fit(x_batch,y_batch)
            except:
                print(str(cv))
        t_time=time()-t_time
        best_params=clf.best_params_
        best_clf = clf.best_estimator_

        write(datasets_name, algo_name,data,test_index,best_clf,t_time,best_params,cv)

if __name__ == '__main__':
    dir_path="datasets/"
    for file in os.listdir(dir_path):
        path=dir_path+file
        main(path)