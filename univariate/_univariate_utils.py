'''
Author   - Sharoon Saxena
Github   - https://github.com/SharoonSaxena
Linkedin - https://www.linkedin.com/in/sharoon95saxena/
e-mail   - saxenasharoon@gmail.com
'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sqrt, abs, round


def descriptive_stats(data):
    """Returns all the descriptives of a numerical variable in a dictionary format containing....

        min
        max
        range
        variance
        standard_deviation
        mean 
        median
        skewness
        kurtosis

    Args:
        data (pandas.Series): numerical column to calculate descriptives for
    """
    stats = {'min': data.min(),
            'max': data.max(),
            'range': data.max()-data.min(),
            'variance': data.var(),
            'standard_deviation': data.std(),
            'mean': data.mean(),
            'median': data.median(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'quant25' : data.quantile(0.25),
            'quant75' : data.quantile(0.75),
            'IQR' : data.quantile(0.75) - data.quantile(0.25),
            'whisker_low' : data.quantile(0.25)-(1.5*(data.quantile(0.75) - data.quantile(0.25))),
            'whisker_high' : data.quantile(0.75)+(1.5*(data.quantile(0.75) - data.quantile(0.25)))}
    return stats

def gaussian_outlier_remover(data, n_std):
    """Removes the outliers from the data using the emperical rule of the gaussian disribution

    Args:
        data (pd.Series): Data colum to apply ooutlier removal on
        n_std (float): will exclude points beyond n_std standard deviations
    """
    avg, std = data.mean(), data.std()
    return data[(data < (avg + (n_std*std))) & (data > (avg - (n_std*std)))]


def whisker_outlier_imputer(data):
    """Imputes the outliers in the data with the upper and lower whiskers+1

    Args:
        data (pd.Series): data colums which needs to be treated
    """
    quant25, quant75 = data.quantile(0.25), data.quantile(0.75)
    IQR = data.quantile(0.75)-data.quantile(0.25)
    whisker_high = quant75 + 1.5*(IQR)
    whisker_low = quant25 - 1.5*(IQR)
    data[data > whisker_high] = whisker_high+1
    data[data < whisker_low] = whisker_low-1
    return data


def num_outlier(data, criteria = 'gaussian', n=3):
    """Returns the number of outliers present in the data based on the criteria chosen.
    Returns 3 value list (low,high,total)

    Args:
        data (pd.Series): data volumn to count the number of outliers in.
        criteria (str, optional): Criteria to calculate number of outliers; 'gaussian' for emperical rule, 'whisker' for whisker criteria. Defaults to 'gaussian'.
        n (int, optional): n times standard deviation to consider, only applicable for 'gaussian'. Defaults to 3.
    """
    if criteria == "gaussian":
        low = len(data[data < (data.mean()-(n*data.std()))])
        high = len(data[data > (data.mean()+(n*data.std()))])
        tot = low+high
        return low, high, tot

    elif criteria == 'whisker':
        low = len(data[data < data.quantile(0.25)-(1.5*(data.quantile(0.75)-data.quantile(0.25)))])
        high = len(data[data > data.quantile(0.75)+(1.5*(data.quantile(0.75)-data.quantile(0.25)))])
        tot = low+high
        return low, high, tot
    else:
        print("mentioned criteria not available, check documentation for available criterion")




'''
Author   - Sharoon Saxena
Github   - https://github.com/SharoonSaxena
Linkedin - https://www.linkedin.com/in/sharoon95saxena/
e-mail   - saxenasharoon@gmail.com
'''
