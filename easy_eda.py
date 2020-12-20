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

def uva_numerical(data, var_group, dpi=100):
    """
    Summary:
    Custom function for easy and efficient analysis of numerical univariate analysis
    
    For any given Numerical variable, this function return 4 plots

    1. KDE plot including outliers
    2. KDE plot excluding outliers
    3. Boxplot including outliers
    4. Boxplot excluding outliers

    Args:
        data (pandas.DataFrame): Data to which variable groups belong to
        var_group (list): list of numerical variables to analyse
        dpi (int > 50) : display pixel density of plots. default=100
    
    """
    
    #looping for each variable
    for i in var_group:
    
      # calculating descriptives of variable
      mini = data[i].min()
      maxi = data[i].max()
      ran = data[i].max()-data[i].min()
      mean = data[i].mean()
      median = data[i].median()
      st_dev = data[i].std()
      skew = data[i].skew()
      kurt = data[i].kurtosis()

      # calculating points of standard deviation
      points = mean-st_dev, mean+st_dev

      #Plotting kde with every informationj+1
      plt.figure(figsize=(25,5), dpi=dpi)       
      plt.subplot(1,4,1)
      sns.kdeplot(data[i], shade=True)
      sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
      sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
      sns.scatterplot([mean], [0], color = 'red', label = "mean")
      sns.scatterplot([median], [0], color = 'blue', label = "median")
      plt.xlabel('{}'.format(i), fontsize = 16)
      plt.ylabel('density', fontsize=16)
      plt.title('Including Outliers\nstd_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format(round(st_dev,2),round(kurt,2),round(skew,2), (round(mini,2),round(maxi,2),round(ran,2)),round(mean,2),round(median,2)),fontsize=14)

      # removing outliers
      tmp = data[i][(data[i]< (mean + (3*st_dev))) & (data[i] > (mean-(3*st_dev)))]
      
      # recalculating necessary statistics
      mini = tmp.min()
      maxi = tmp.max()
      ran = tmp.max()-tmp.min()

      #Plotting kde (without outliers) with every information
      plt.subplot(1,4,2)
      sns.kdeplot(tmp, shade=True)
      sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
      sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
      sns.scatterplot([mean], [0], color = 'red', label = "mean")
      sns.scatterplot([median], [0], color = 'blue', label = "median")
      plt.xlabel('{}'.format(i), fontsize = 16)
      plt.ylabel('density', fontsize=16)
      plt.title('Excluding Outliers\nstd_dev = {}; kurtosis = {};\nskew = {}; range u= {}\nmean = {}; median = {}'.format(round(st_dev,2),
                round(kurt,2),
                round(skew,2),
                (round(mini,2),round(maxi,2),round(ran,2)),
                round(mean,2),
                round(median,2)), fontsize=14)

      
      ## box plot with outliers
      # calculating descriptives of variable
      quant25 = data[i].quantile(0.25)
      quant75 = data[i].quantile(0.75)
      IQR = quant75 - quant25
      med = data[i].median()
      whis_low = quant25-(1.5*IQR)
      whis_high = quant75+(1.5*IQR)

      # Calculating Number of Outliers
      outlier_high = len(data[i][data[i]>whis_high])
      outlier_low = len(data[i][data[i]<whis_low])

      #Plotting the variable with every information
      plt.subplot(1,4,3)
      sns.boxplot(data[i], orient="v")
      plt.ylabel('{}'.format(i), fontsize=16)
      plt.title('Including Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                         round(IQR,2),
                         round(med,2),
                         (round(quant25,2),round(quant75,2)),
                         (outlier_low,outlier_high)
                         ),fontsize=14)
      
      ## box plot without outliers  
      # replacing outliers with max/min whisker
      tmp = data[i][:]
      tmp[tmp>whis_high] = whis_high+1
      tmp[tmp<whis_low] = whis_low-1
      
      # plotting without outliers
      plt.subplot(1,4,4)
      sns.boxplot(tmp, orient="v")
      plt.ylabel('{}'.format(i), fontsize=16)
      plt.title('Excluding Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                         round(IQR,2),
                         round(med,2),
                         (round(quant25,2),round(quant75,2)),
                         (outlier_low,outlier_high)
                         ),fontsize=14)
      plt.tight_layout()
      plt.show()









def uva_categorical(data, var_group, max_uni=5, dpi=100, include_missing=False):
    """Custom function for easy visualisation/analysis of Categorical Variables

    Args:
        data (pandas.DataFrame): Data to which variable groups belong to
        var_group (list): list of categorical variables to analyse
        max_uni (int > 0) : to show frequency table only if unique categories less than max_uni
        dpi (int > 50) : display pixel density of plots. default=100
        include_missing (bool) : whether to include missing values or not. default = False
    """
    # for each variable in var_group
    for cat in var_group:

        # setting plot size checking for table
        plt.figure(figsize=(12,5), dpi=100)

        # to print table or not
        if data[cat].nunique()>max_uni:
            table=False
        else:
            table=True

        # absolute value bar graph
        plt.subplot(1,2,1)

        # include missing values?
        if include_missing==False:
            tmp = data[cat].value_counts(normalize=True).map(lambda x:round(x*100,3))
            title1 = '%frequency\n{}'.format(tmp)
            title2 = 'Categories = {}'.format(len(tmp))
        else:
            tmp = data[cat].value_counts(dropna=False, normalize=True).map(lambda x:round(x*100,3))
            title1 = '%frequency (including missing if any)\n{}'.format(tmp)
            title2 = 'Categories (including missing if any)\n{}'.format(len(tmp))

        # plotting
        plt.barh([str(i) for i in tmp.index], tmp)

        # checking whether table needed
        if table==True:
            plt.title(title1, fontsize=14)
        else:
            plt.title(title2, fontsize=14)

        # pie chart with relative frequency
        plt.subplot(1,2,2)

        tmp = tmp.map(lambda x:round(x*100,3))
        tmp.plot.pie()
        if table==True:
            plt.title(title1, fontsize=14)
        else:
            plt.title(title2, fontsize=14)
        plt.tight_layout()
        plt.show()






    


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








def Hypothesis_Testing_cont_cat(data, cont, cat, category, tester='T', alpha=0.05, dpi=100):
    """
    This function performs 2-Tail test for Numerical and Categorical Bivariate Analysis.\n
       Also Returns the following graphs:

       1. Sample/Population Mean Barplot and Result of Hypothesis Testing
       2. Sample/Population Boxplot distribution
       3. Sample/Population Kde distribution

    Args:
        data (pandas.dataframe): Dataframe which contains all the variables
        cont (string): Column name of the continuous variable
        cat (string): Column name of the categorical variable
        category (string): Name of the sample/category in the categorical column
        tester (str, optional): Tester you want to use (Z-test or T-Test). Defaults to 'T'.
        alpha (float, optional): Significance level you want to choose, p-value will be benchmarked agains this. Defaults to 0.05.
    """
    #creating 2 samples
    x1 = data[cont][data[cat]==category][:]
    x2 = data[cont][~(data[cat]==category)][:]
  
    #calculating descriptives
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.mean()
  
    #calculating p-values
    if tester=='T':
      p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
    else:  
      p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

    # is p-val significant?
    if p_val<alpha:
      sig = True
    else:
      sig= False
  
    #table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

    #plotting
    plt.figure(figsize = (18,5), dpi=dpi)
  
    #barplot, hyp-test and categorywise mean
    plt.subplot(1,3,1)
    sns.barplot([str(category),'not {}'.format(category)], [m1, m2])
    plt.ylabel('mean {}'.format(cont),fontsize=14)
    plt.xlabel(cat, fontsize=14)
    plt.title('Category-wise Mean\np-value = {}\n Difference Significant? = {} \n{}'.format(round(p_val,5), sig, table), fontsize=14)

    # boxplot including outliers
    plt.subplot(1,3,2)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('Categorical Distribution',fontsize=14)
    plt.ylabel(cat,fontsize=14)
    # category-wise distribution
    plt.subplot(1,3,3)
    sns.kdeplot(x1, shade= True, color='blue', label = str(category))
    sns.kdeplot(x2, shade= False, color='green', label = 'not {}'.format(category), linewidth = 1)
    plt.title('categorical distribution', fontsize=14)
    plt.tight_layout()













'''
Author   - Sharoon Saxena
Github   - https://github.com/SharoonSaxena
Linkedin - https://www.linkedin.com/in/sharoon95saxena/
e-mail   - saxenasharoon@gmail.com
'''