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
from CL import write


def main(path):
    """
    The main function
    :param path: Path to the DS
    :type path: String
    """
    algo_name="RF"
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