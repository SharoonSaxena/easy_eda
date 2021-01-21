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
from ._univariate_utils import *






def numerical_summary(data, var_group, dpi=100):
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
      stats = descriptive_stats(data[i])

      # calculating points of standard deviation
      points = stats['mean'] - stats['standard_deviation'], stats['mean'] + stats['standard_deviation']

      #calculating number of outliers
      outlier_low, outlier_high, outlier_total = num_outlier(data[i],criteria='gaussian',n=3)

      #Plotting kde with every informationj+1
      plt.figure(figsize=(25,5), dpi=dpi)       
      plt.subplot(1,4,1)
      sns.kdeplot(data[i], shade=True)
      sns.lineplot(points, [0,0], color = 'black', label = "std_dev", dashes=True, linewidth=3)
      sns.scatterplot([stats['min'],stats['max']], [0,0], color = 'orange', label = "min/max", s=100)
      sns.scatterplot([stats['mean']], [0], color = 'red', label = "mean", s=100)
      sns.scatterplot([stats['median']], [0], color = 'blue', label = "median", s=100)
      plt.xlabel('{}'.format(i), fontsize = 16)
      plt.ylabel('density', fontsize=16)
      plt.title('Including Outliers\nstd_dev = {}; kurtosis = {};\nskew = {}; (min,max,range) = {}\nmean = {}; median = {}\n outliers (low,high,total) = {}'.format(
          round(stats['standard_deviation'],2),
          round(stats['kurtosis'],2),
          round(stats['kurtosis'],2), 
          (round(stats['min'],2),
          round(stats['max'],2),
          round(stats['range'],2)),
          round(stats['mean'],2),
          round(stats['median'],2),
          (outlier_low,outlier_high,outlier_total)),
          fontsize=14)


      # removing outliers
      tmp = gaussian_outlier_remover(data[i],3)
      
      # recalculating necessary statistics
      tmp_stats = descriptive_stats(tmp)
      

      #Plotting kde (without outliers) with every information
      plt.subplot(1,4,2)
      sns.kdeplot(tmp, shade=True)
      sns.lineplot(points, [0,0], color = 'black', label = "std_dev", dashes=True, linewidth=3)
      sns.scatterplot([tmp_stats['min'],tmp_stats['max']], [0,0], color = 'orange', label = "min/max", s=100)
      sns.scatterplot([tmp_stats['mean']], [0], color = 'red', label = "mean", s=100)
      sns.scatterplot([tmp_stats['median']], [0], color = 'blue', label = "median", s=100)
      plt.xlabel('{}'.format(i), fontsize = 16)
      plt.ylabel('density', fontsize=16)
      plt.title('Excluding Outliers\nstd_dev = {}; kurtosis = {};\nskew = {}; (min,max,range) = {}\nmean = {}; median = {}\n outliers_REMOVED (low,high,total) = {}'.format(
          round(tmp_stats['standard_deviation'],2),
          round(tmp_stats['kurtosis'],2),
          round(tmp_stats['skewness'],2), 
          (round(tmp_stats['min'],2),
          round(tmp_stats['max'],2),
          round(tmp_stats['range'],2)),
          round(tmp_stats['mean'],2),
          round(tmp_stats['median'],2),
          (outlier_low,outlier_high,outlier_total)),
          fontsize=14)

      
      ## box plot with outliers
    
      # Calculating Number of Outliers
      outlier_low, outlier_high, outlier_total = num_outlier(data[i], criteria='whisker')

      #Plotting the variable with every information
      plt.subplot(1,4,3)
      sns.boxplot(data[i], orient="v")
      plt.ylabel('{}'.format(i), fontsize=16)
      plt.title('Including Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low,high,total) = {} \n'.format(
                         round(stats['IQR'],2),
                         round(stats['median'],2),
                         (round(stats['quant25'],2),round(stats['quant75'],2)),
                         (outlier_low,outlier_high, outlier_total )
                         ),fontsize=14)
      
      ## box plot with imputed outliers  
      # replacing outliers with max/min whisker
      tmp = whisker_outlier_imputer(data[i])
      
      # plotting without outliers
      plt.subplot(1,4,4)
      sns.boxplot(tmp, orient="v")
      plt.ylabel('{}'.format(i), fontsize=16)
      plt.title('With Imputed Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low,high,total) = {} \n'.format(
                         round(stats['IQR'],2),
                         round(stats['median'],2),
                         (round(stats['quant25'],2),round(stats['quant75'],2)),
                         (outlier_low,outlier_high,outlier_total)
                         ),fontsize=14)
      plt.tight_layout()
      plt.show()









def categorical_summary(data, var_group, max_uni=5, dpi=100, include_missing=False):
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

        

        # to print table or not
        if data[cat].nunique()>max_uni:
            table=False
        else:
            table=True

        # setting plot size checking for table
        if table==True:
            plt.figure(figsize=(12,6), dpi=100)
        else:
            plt.figure(figsize=(12,5), dpi=100)
            

        # absolute value bar graph
        plt.subplot(1,2,1)

        # include missing values?
        if include_missing==False:
            tmp = data[cat].value_counts(normalize=True).map(lambda x:round(x*100,3))
            title1 = '%frequency\n{}'.format(tmp)
            title2 = 'Categories\n{}'.format(len(tmp))
        else:
            tmp = data[cat].value_counts(dropna=False, normalize=True).map(lambda x:round(x*100,3))
            title1 = '%frequency (including missing if any) \n{}'.format(tmp)
            title2 = 'Categories (including missing if any) \n{}'.format(len(tmp))

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




'''
Author   - Sharoon Saxena
Github   - https://github.com/SharoonSaxena
Linkedin - https://www.linkedin.com/in/sharoon95saxena/
e-mail   - saxenasharoon@gmail.com
'''