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
      plt.title('Excluding Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low,high,total) = {} \n'.format(
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
            title1 = '%frequency = {}\n'.format(tmp)
            title2 = 'Categories = {}\n'.format(len(tmp))
        else:
            tmp = data[cat].value_counts(dropna=False, normalize=True).map(lambda x:round(x*100,3))
            title1 = '%frequency (including missing if any) = {}\n'.format(tmp)
            title2 = 'Categories (including missing if any) = {}\n'.format(len(tmp))

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