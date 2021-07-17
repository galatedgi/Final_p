import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def get_avg(measure):
    """
    This function calculates an average for a measure per DS and algorithms
    :param measure:The measure for which to calculate an average
    :type measure: String
    :return: Data Frame containing the averages for the measure per DS and algorithm
    :rtype: Data Frame
    """
    algo=["CL","CL_s","RF"]
    result=pd.DataFrame(index=algo)
    df=pd.read_csv("results.csv")
    for file in os.listdir("datasets/"):
        avg=[]
        for a in algo:
            df_exp=df.loc[(df['Dataset Name'] == file) & (df['Algorithm Name'] == a)]
            mean=df_exp[measure].mean()
            avg.append(mean)
        result.insert(0,file,avg)
    return result.transpose()

def plot_graph(measure):
    """
    This function creates a graph of the averages for the measure per DS and algorithm
    :param measure:The measure for which to creates a graph
    :type measure:String
    """
    result=get_avg(measure)
    algo=["CL","CL_s","RF"]
    i=0
    marker = ['o', 'x', '+']
    ticks = os.listdir("datasets/")
    x=np.array(list(range(1,len(ticks)+1)))
    plots=[]
    for a in algo:
        data=result[a]
        plot=plt.scatter(x,np.array(data),marker=marker[i])
        plots.append(plot)
        i+=1
    plt.xlabel("#Dataset")
    plt.ylabel(measure)
    plt.legend(plots,algo,loc='lower left')
    plt.xticks(x)
    plt.show()


if __name__ == '__main__':
    plot_graph("Inference Time")