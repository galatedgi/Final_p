import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


if __name__ == '__main__':
    # Fridman Test
    avg=[]
    algo=["CL","CL_s","RF"]
    df=pd.read_csv("results.csv")
    for file in os.listdir("datasets/"):
        avg3=[]
        for a in algo:
            df_exp=df.loc[(df['Dataset Name'] == file) & (df['Algorithm Name'] == a)]
            mean=df_exp['AUC'].mean()
            avg3.append(mean)
        avg.append(avg3)


    # data=np.array(df)
    # new_data = data.drop('datasetName', axis=1)
    # groups = []
    # for i in range(new_data.shape[0]):
    #     group = new_data.iloc[i, :]
    #     group_arr = [group[0], group[1], group[2]]
    #     groups.append(group_arr)

    stat, p = friedmanchisquare(*avg)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05

    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
        # hoc
        print(sp.posthoc_nemenyi_friedman(avg))

