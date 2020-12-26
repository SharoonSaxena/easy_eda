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
from scipy.stats import norm
from scipy.stats import t as t_dist


def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    """Calculates 2 Sample Z-test, Returns p-value
      Args:
          X1 (float): Mean of sample-1
          X2 (float): Mean of sample-2
          sd1 (float): Standard deviation of sample-1
          sd2 (float): Standard deviation of sample-2
          n1 (int): Number of instances in sample-1
          n2 ([int]): Number of instances in sample-2

      Returns:
          [float]: p-value for Z-test
    """
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval









def TwoSampT(X1, X2, sd1, sd2, n1, n2):
    """Calculates 2 Sample T-test, Returns p-value
      Args:
          X1 (float): Mean of sample-1
          X2 (float): Mean of sample-2
          sd1 (float): Standard deviation of sample-1
          sd2 (float): Standard deviation of sample-2
          n1 (int): Number of instances in sample-1
          n2 ([int]): Number of instances in sample-2

      Returns:
          [float]: p-value for T-test 
    """
    ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
    t = (X1 - X2)/ovr_sd
    df = n1+n2-2
    pval = 2*(1 - t_dist.cdf(abs(t),df))
    return pval
    








def bva_cont_cat(data, cont, cat, print_table=True):
    """Returns the complete Bivariate statistics and plots of Numerical and Categorical combination
        This function returns the following graphs:
        1. Categorywise mean barplot
        2. Categorywise median barplot
        3. Categorywise distribution using Boxplot
    Args:
        data (pandas.Dataframe): Dataframe which contains the continuous and categorical variable
        cont (string): Name of the continuous variable
        cat ([type]): Name og the Categorical variable
        print_table (bool, optional): Whether to print groupy table as title or not. Defaults to True.
    """

    # calculating categorywise mean and median
    tmp1 = data.groupby(cat)[cont].mean().map(lambda x: round(x,3))
    tmp2 = data.groupby(cat)[cont].median().map(lambda x: round(x,3))

    # setting figuresize
    plt.figure(figsize = (18,5), dpi = 100)

    # categorywise mean barplot
    plt.subplot(1,3,1)
    plt.bar(tmp1.index[:], tmp1[:])
    plt.xlabel('{}'.format(cat),fontsize=14)
    plt.ylabel('Mean', fontsize=14)
    if print_table==True:
        plt.title("{}".format(tmp1))
    else:
        plt.title("Mean age w.r.t education", fontsize=14)

    # categorywise median barplot
    plt.subplot(1,3,2)
    plt.bar(tmp2.index[:], tmp2[:])
    plt.xlabel('{}'.format(cat),fontsize=14)
    plt.ylabel('Median', fontsize=14)
    if print_table==True:
        plt.title("{}".format(tmp2))
    else:
        plt.title("Median age w.r.t education", fontsize=14)

    # categorywise distribution boxplot
    plt.subplot(1,3,3)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title("{} distribution w.r.t. {}".format(cont,cat), fontsize=14)
    plt.xlabel('{}'.format(cat),fontsize=14)
    plt.ylabel('{}'.format(cont), fontsize=14)

    plt.tight_layout()
    plt.show()     









def bva_cat_cat(data, cat1, cat2, print_table=True):
    """generate plots to summarise categorical categorical bivariate analysis

    Args:
        data (pandas.dataframe): dataframe which contains the categorical variables
        cat1 (string): name of categorical column1
        cat2 ([type]): name of categorical column2
        print_table (bool, optional): whether to print crosstab or not. Defaults to True.

    Returns:
        [pd.dataframe]: returns a cross tab only if print_table=True
    """

    # calculating crosstab, columns>rows
    if data[cat1].nunique()>=data[cat2].nunique():
        tmp = pd.crosstab(index=data[cat2], columns=data[cat1], normalize=True)
        tmpr = pd.crosstab(index=data[cat1], columns=data[cat2], normalize=True)
    else:
        tmp = pd.crosstab(index=data[cat1], columns=data[cat2], normalize=True)
        tmpr = pd.crosstab(index=data[cat2], columns=data[cat1], normalize=True)

    # setting figure size
    _, axes = plt.subplots(2,2, figsize=(12,12), dpi=150)
    
    # Grouped Bar plot1
    p1 = tmp.plot(kind ='bar', ax=axes[0,0])
    p1.set_title("{} grouped w.r.t. {}".format(cat2,cat1), fontsize=14)
    p1.set_ylabel('% frequency', fontsize=14)
    p1.set_xlabel(cat1, fontsize=14)
    
    # stacked bar plot1
    p2 = tmp.plot.bar(stacked=True, ax=axes[0,1])
    p2.set_ylabel('% frequency', fontsize=14)
    p2.set_title("{} stacked w.r.t {}".format(cat2,cat1), fontsize=14)
    p2.set_xlabel(cat1, fontsize=14)

    # Grouped Bar plot2
    p3 = tmpr.plot(kind ='bar', ax=axes[1,0])
    p3.set_title("{} grouped w.r.t {}".format(cat1,cat2), fontsize=14)
    p3.set_ylabel('% frequency', fontsize=14)
    p3.set_xlabel(cat2, fontsize=14)
    
    # stacked bar plot2
    p4 = tmpr.plot.bar(stacked=True, ax=axes[1,1])
    p4.set_ylabel('% frequency', fontsize=14)
    p4.set_title("{} stacked w.r.t {}".format(cat1,cat2), fontsize=14)
    p4.set_xlabel(cat2, fontsize=14)

    plt.tight_layout()
    plt.show()
    
    # if crosstab needs to be printed
    if print_table==True:
        return tmp



'''
Author   - Sharoon Saxena
Github   - https://github.com/SharoonSaxena
Linkedin - https://www.linkedin.com/in/sharoon95saxena/
e-mail   - saxenasharoon@gmail.com
'''