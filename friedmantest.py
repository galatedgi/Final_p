import os
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


def friedman_test():
    """
    This function is the friedman test
    """
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

    stat, p = friedmanchisquare(*avg)
    print('Statistics=%.10f, p=%.10f' % (stat, p))

    alpha = 0.05

    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
        # hoc
        hoc_test=sp.posthoc_nemenyi_friedman(avg)
        col_name={}
        for a in algo:
            col_name.update({algo.index(a):a})
        hoc_test.rename(columns=col_name,index=col_name,inplace = True)
        print(hoc_test)


if __name__ == '__main__':
    friedman_test()

