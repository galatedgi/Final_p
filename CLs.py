import os
from time import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from tqdm import tqdm
from CL import svm_score,get_data_by_rank,write




def CL_s(data,score_func=svm_score):
    """
    This function is the implement the improvement for the CL algorithm from the paper
    :param data:The dataset
    :type data:ndarray
    :param score_func:The score function
    :type score_func:function
    :return:The data batches
    :rtype:list of ndarray
    """
    rank=score_func(data)
    new_data=get_data_by_rank(data,rank)
    return new_data

def main(path):
    """
    The main function
    :param path: Path to the DS
    :type path: String
    """
    algo_name="CL_s"
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
        r_data=CL_s(data[train_index])
        clf = RandomForestClassifier()
        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0, random_state=42)
        t_time=time()
        rank_data=np.array(r_data)
        y_train=rank_data[:, -1:]
        x_train= rank_data[:, :-1]
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

