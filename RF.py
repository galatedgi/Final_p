import csv
import os
from time import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from tqdm import tqdm


def write(datasets_name, algo_name,data,test_index,best_clf,t_time,best_params,cv):
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
    algo_name="RF"
    # path="datasets/chess-krvkp.csv"
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
        train=data[train_index]
        clf = RandomForestClassifier()
        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0, random_state=42)
        t_time=time()
        y_train=train[:, -1:]
        x_train= train[:, :-1]
        clf.fit(x_train,y_train)
        t_time=time()-t_time
        best_params=clf.best_params_
        best_clf = clf.best_estimator_

        write(datasets_name, algo_name,data,test_index,best_clf,t_time,best_params,cv)


if __name__ == '__main__':
    dir_path="datasets/"
    for file in os.listdir(dir_path):
        path=dir_path+file
        main(path)