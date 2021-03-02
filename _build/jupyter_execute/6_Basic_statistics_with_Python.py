## Basic statistics and time series analysis with Python
This chapter shortly introduces the methods and libraries Python has to offer for basic statistics and time series analysis.

Let's start by analysing what kind statistical functions Numpy has to offer. The selection is somewhat limited and usually it is better to use some other libraries for statistical analysis, like Pandas or Statsmodels. However, sometimes you have to use Numpy for data manipulations. For example, if your accounting datasets are very large, Numpy is the most efficient solution in Python. Therefore, it is also useful to know the  basic statistical functions that Numpy has to offer. We will learn more about Numpy in the next chapter. Here we check only the main statistical functions.

import numpy as np
import matplotlib.pyplot as plt
plt.xkcd()
#plt.style.use('bmh')


The following function creates an array of random values from the normal distribution. The mean of the distribution is 0 and the standard deviation (scale) 1. The last parameter (5,3) defines the shape of the array.

array_np = np.random.normal(0,1,(5,3))

array_np

The default is a statistic calculated from the whole array, but you can also define if you want them calculated for rows or colums.

array_np.mean()  # The mean for the whole array

array_np.mean(0) # The mean for the columns

array_np.mean(1) # The mean for the rows

Numpy also has the function to calculate the sum of the values.

array_np.sum()

Standard deviation

array_np.std()

*argmin* and *argmax* return the position of the minimum and maximum value. By default, *argmin* and *argmax* return the index of a flattened array (transformed to one dimension). You can also define the axis.

array_np.argmin() # The position of the smallest value.

array_np.argmin(0) # The position of the largest value in columns.

array_np.argmin(1) # The position of the largest value in rows.

Cumulative sum of the array. **cumsum()** also flattens arrays by default.

array_np.cumsum()

plt.plot(array_np.cumsum())
plt.show()

### Descriptive statistics with Pandas

Pandas is much more versatile for statistical calculations than Numpy, and should be used if there is no specific reason to use Numpy. Let's load a more interesting dataset to analyse.

import pandas as pd

stat_df = pd.read_csv('stat_data.csv',index_col=0)

stat_df

stat_df.set_index('NAME',inplace=True) # Set the NAME variable as index of the dataframe

Usually, the default setting with the Pandas statistical functions is that they calculate column statistics. For example, here is the **sum** function (does not make much sense here).

stat_df.sum()

With **axis=1**, you can calculate also row-sums.

stat_df.sum(axis=1)

There is also a cumulative sum similar to Numpy's equivalent. Notice that it calculates the cumulative sum series for columns by default.

stat_df[['DIV. YIELD','ROE (%)']].cumsum()

The mean value of the columns. Notice how non-numerical columns are automatically excluded.

stat_df.mean()

By default, NA values are excluded. You can prevent that using **skipna=False**.

stat_df.mean(skipna=False)

**idxmin** and **idxmax** can be used to locate the maximum and minimum values along the specified axis. It does not work with string-values, so we restrict the columns.

stat_df[['DIV. YIELD', 'ROE (%)', 'R&D/SALES (%)',
       'CoGS/SALES - 5 Y (%)', 'SG%A/SALES 5Y (%)']].idxmax()

**Describe()** is the main tool for descriptive statistics.

stat_df.describe()

You can also use **describe** for string-data.

stat_df['ACCOUNTING STANDARD'].describe()

Quick histograms are easy to draw with Pandas **hist()**.

import matplotlib.pyplot as plt

stat_df.hist(figsize=(10,10),grid=False,edgecolor='k')
plt.show()

It is also easy to draw bar plots, line plots, etc. from variables.

stat_df.iloc[0:10]['ROE (%)'].plot.bar(grid=False,edgecolor='k')
plt.show()

Pandas also has functions for quantiles, median, mean absolute deviation, variance, st. dev., skewness, kurtosis etc.

stat_df.median()

stat_df.std()

stat_df.skew()

stat_df.kurt()

stat_df.mad() # Mean absolute deviation

There are also functions for first differences (**diff**) and percentage changes (**pct_change**) that are useful for time series. We check them more closely when we discuss time series.

A correlation matrix can be calculated with **corr()**.

stat_df.corr()

Seaborn can be used to visualise the correlation matrix.

import seaborn as sns

sns.heatmap(stat_df.corr())
plt.show()

There is also a function for covariance.

stat_df.cov()

If you want to calculate correlations between two dataframes, you can use **corr_with()**.

To collect the unique values of a Pandas series, you can use **unique()**.

stat_df['IBES COUNTRY CODE'].unique()

**value_counts** can be used to collect frequencies of values. Here the counts are presented as a bar chart. Notice how we chain dataframe functions.

stat_df['IBES COUNTRY CODE'].value_counts().plot.bar(figsize=(12,8),grid=False)
plt.show()

### Probability and statistics functions

What we mainly need in data analysis from probability theory are random variables and distributions. Numpy has a large collection of random number generators that are located in module **numpy.random**. There are random number generators for every distribution that you will ever need.

x_np = np.random.normal(size=500) # Create 500 normal random variables with *mean=0* and *std=1*
y_np = np.random.normal(size=500)
plt.hist2d(x_np,y_np,bins=30) # Visualise the joint frequency distribution using 2D-histogram.
plt.show()

hyper_np = np.random.hypergeometric(8,9,12,size=(3,3))

hyper_np

Probably the most important distributions are the standard normal distribution, the chi2 distribution and the binomial distribution.

plt.hist(np.random.normal(size=2000),bins=20,color='lightgray',edgecolor='k')
plt.show()

plt.hist(np.random.chisquare(1,size=1000),bins=20,color='lightgray',edgecolor='k')
plt.show()

plt.hist(np.random.chisquare(6,size=1000),bins=20,color='lightgray',edgecolor='k')
plt.show()

sns.countplot(x=np.random.binomial(20,0.7,size=1000),color='lightgray',edgecolor='k')
plt.show()

It is good to remember that computer-generated random numbers are not truly random numbers. They are so called pseudorandom numbers. It is because they are generated using a deterministic algorithm and a seed value.

### Statistical analysis with statsmodels

Pandas (and Numpy) has only functions for basic statistical analysis, like descriptive statistics. If you want to do more advanced (traditional) statistical analysis, the **statsmodels** library is a good option.

For example, linear regerssion models are very easy to build with **statsmodels**. Let's model the dividend yield as a function of the R&D intensity.

import statsmodels.api as sm
import seaborn as sns

Remove the missing values of the endogenous variable.

stat_df.columns

reduced_df = stat_df[~stat_df['DIV. YIELD'].isna()]

reduced_df

One curiosity with statsmodels is that you need to add constant to the x-variables.

x = sm.add_constant(reduced_df['R&D/SALES (%)'])

model = sm.OLS(reduced_df['DIV. YIELD'],x,missing='drop')

results = model.fit()

The parameters and t-values of the model.

results.params

results.tvalues

With **summary()**, you can output all the key regression results.

results.summary()

results.summary2() # Different format for the results.

**Regplot** can be usd to plot the model over observations.

sns.regplot(x='R&D/SALES (%)',y='DIV. YIELD',data=reduced_df)
plt.show()

It is easy to add dummy-variables to your model using the Pandas **get_dummes()** -function. For more than two categories, it creates more than one dummmy-variables. 

Create a new variable **acc_dummy** that has a value one if the company has accounting controversies.

reduced_df['acc_dummy'] = pd.get_dummies(stat_df['Accounting Controversies'],drop_first=True)

x = sm.add_constant(reduced_df[['R&D/SALES (%)','acc_dummy']])

Let's chance also the dependent  variable to **ROE**.

model = sm.OLS(reduced_df['ROE (%)'],x,missing='drop')

results = model.fit()

results.summary()

As you can see, there is something wrong. If you observed the histograms carefully, you'd seen that there is a clear outlier in **ROE** values.

reduced_df['ROE (%)'].hist(grid=False)

reduced_df['ROE (%)'].describe()

A maximum value of 31560! We can remove it, or we can winsorise the data. Let's winsorise.

reduced_df['ROE (%)'].clip(lower = reduced_df['ROE (%)'].quantile(0.025),
                           upper = reduced_df['ROE (%)'].quantile(0.975),inplace=True)

reduced_df['ROE (%)'].hist(grid=False)
plt.show()

Let's try to build the regression model again.

x = sm.add_constant(reduced_df[['R&D/SALES (%)','acc_dummy']])

model = sm.OLS(reduced_df['ROE (%)'],x,missing='drop')

results = model.fit()

results.summary()

Although accounting controversies has a negative coefficient, it is not statisticall significant.

Statsmodels is a very comprehensive statistical library. It has modules for nonparametric statistics, generalised linear models, robust regression and time series analysis. We do not go into more details of Statsmodels at this point. If you want to learn more, check the Statsmodels documentation: [www.statsmodels.org/stable/index.html](https://www.statsmodels.org/stable/index.html)

### Statistical analysis with scipy.stats

Another option for statistical analysis in Python is the **stats** module of **Scipy**. It has many functions that are not included in other statistical libraries. However, Scipy does not handle automatically nan-values. Accounting data almost always has missing values, thus, they need to be manually handled, which is a bit annoying.

Like Numpy, it has an extensive range of random number generators and probability distributions. It also has a long list of statistical tests.

import scipy.stats as ss

In SciPy, random variables with different distributions are presented as classes, which have methods for random number generation, computing the PDF, CDF and inverse CDF, fitting parameters and computing moments.

norm_rv = ss.norm(loc=1.0, scale=0.5)

Expected value

norm_rv.expect()

The value of probability distribution function at 0.

norm_rv.pdf(0.)

The value of cumulative distribution function at 1.

norm_rv.cdf(1.)

Standard deviation

norm_rv.std()

The visualisation of the pdf and cdf of the RV.

plt.plot(np.linspace(-1,3,100),[norm_rv.pdf(x) for x in np.linspace(-1,3,100)])
plt.plot(np.linspace(-1,3,100),[norm_rv.cdf(x) for x in np.linspace(-1,3,100)],'r--')
plt.show()

The list of distributions in Scipy is long. Here you can read more about them: [docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)

There are many functions for descriptive statistics. The full list can be found also from the link above.

ss.describe(stat_df['DIV. YIELD'])

ss.describe(stat_df['ROE (%)'],nan_policy='omit') # Omit missing values

With trimmed mean, we can define the limits to the values of a variable from which the mean is calculated. There are **trimmed** versions for many other statistics too.

ss.tmean(stat_df['ROE (%)'].dropna(),limits=(-50,50))

Standard error of the mean.

ss.sem(stat_df['DIV. YIELD'])

Bayesian confidence intervals.

ss.bayes_mvs(stat_df['DIV. YIELD'])

Interquantile range

ss.iqr(stat_df['DIV. YIELD'])

The list of correlation functions is also extensive. The functions also return the p-value from the signficance test.

temp_df = stat_df[['DIV. YIELD','SG%A/SALES 5Y (%)']].dropna()

The output of SciPy is a bit ascetic. The first value is the correlation coefficient and the second value is the p-value.

Pearson's correlation coefficient

ss.pearsonr(temp_df['DIV. YIELD'],temp_df['SG%A/SALES 5Y (%)']) 

Spearman's rank correlation coefficient

ss.spearmanr(temp_df['DIV. YIELD'],temp_df['SG%A/SALES 5Y (%)'])

There are many statistical tests included. Let's divide our data to US companies and others to test the Scipy **ttest()**

us_df = stat_df[stat_df['IBES COUNTRY CODE'] == '  US'] # US
nonus_df = stat_df[~(stat_df['IBES COUNTRY CODE'] == 'US')] # Non-US (notice the tilde symbol)

This test assumes equal variance for both groups. The result implies that the (mean) dividend yield is higher in non-US companies.

ss.ttest_ind(us_df['DIV. YIELD'],nonus_df['DIV. YIELD'],nan_policy='omit')

There are functions for many other statistical tasks, like transformations, statistical distnaces, contigency tables. Check the Scipy homepage for more details.

### Time series

Time series analysis is an important topic in accounting. Python and Pandas has many functions for time series analysis. Time series data has usually fixed frequency, which means that data points occur at regular intervals. Time series can also be irregular, which can potentially make the analysis very difficult. Luckily, Python/Pandas simplifies things considerably.

#### Datetime

Python has modules for date/time -handling by default, the most important being **datetime**.

from datetime import datetime

datetime.now() # Date and time now

datetime.now().year

datetime.now().second

You can calculate with the datetime objects.

difference = datetime(2020,10,10) - datetime(1,1,1)

difference

difference.days

You can use **timedelta** to transform datetime objects.

from datetime import timedelta

date1 = datetime(2020,1,1)

The first argument of timedelta is days.

date1 + timedelta(12)

Dates can be easily turned into string using the Python **str()** function.

str(date1)

If you want to specify the date/time -format, you can use the strftime method of the datetime object.

date2 = datetime(2015,3,18)

%Y: four digit year, %m: two-digit month, %d: two-digit day, %W: week number (Monday is the first day of a week.)

date2.strftime('%Y - %m - %d : Week number %W')

There is also an opposite method, **strptime()**, that turns a string into a datetime object.

sample_data = 'Year: 2012, Month: 10, Day: 12'

datetime.strptime(sample_data, 'Year: %Y, Month: %m, Day: %d')

As you can see, you can strip the date information efficiently, if you know the the format of your date-string. There is also a great non-standard library that can be used to automatically strip date from many different date representations.

from dateutil.parser import parse

parse('Dec 8, 2009')

parse('11th of March, 2018')

You have to be careful with the following syntax.

parse('8/3/2015')

parse('8/3/2015',dayfirst = True)

The most important Pandas function for handling dates is **to_datetime**. We will see many applications of it in the following, but let's first load an interesting time series.

The data contains accountants and auditors as a percent of the US labor force 1850 to 2016. This data is a little bit difficult, because the frequency is irregular. The first datapoints have a ten-year interval, and from 2000 onwards the interval is one year.

times_df =pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/AccountantsAuditorsPct.csv')

times_df.head()

times_df.rename({'Unnamed: 0':'Year'},axis=1,inplace=True) # Correct the name of the year variable

times_df.head()

With **to_datetime**, we can transform the years as datetime objects.

times_df['Year'] = pd.to_datetime(times_df['Year'],format = '%Y')

times_df.head()

Plotting and other things are much easier, if we set the dates as the index.

times_df.set_index('Year',inplace=True)

There is a special object in Pandas for indices that have datetime objects, **Datetimeindex**.

type(times_df.index)

You can pick up the timestamp from an index value.

time_stamp = times_df.index[8]

time_stamp

Now you can pick up values with these timestamps.

times_df.loc[time_stamp]

Actually, you can also use date strings. Pandas will automatically transform it.

times_df.loc['1980']

Slicing works also.

times_df.loc['2010':]

Plotting is easy with the Pandas' built-in functions.

times_df.plot()
plt.show()

Everything works nicely, although we have a dataset with irregular frequency.

If we want, we can transform the series into a fixed frequency time series using **resample**..

times_df.head(15)

Using yearly resampling adds missing values, because for most of the interval we are increasing the frequency.

times_df.resample('Y').mean().head(15)

times_df.resample('Y').mean().plot(style = '.')
plt.show()

If we want, we can fill the missing values with **fillna**.

times_df.resample('Y').ffill().head(15)

times_df.resample('Y').ffill().plot()
plt.show()

We can also decrease the frequency and decide how the original data is aggregated. Let's simulate a stock price using a random walk series.

sample_period = pd.date_range('2010-06-1', periods = 500, freq='D')

temp_np = np.random.randint(0,2,500)

temp_np = np.where(temp_np > 0,1,-1)

rwalk_df = pd.Series(np.cumsum(temp_np), index=sample_period)

rwalk_df.head(15)

Resample as weekly data where the daily values are aggregated together with **mean()**.

rwalk_df.resample('W').mean()

rwalk_df.plot(linewidth=0.5)
rwalk_df.resample('W').mean().plot(figsize=(10,10))
plt.show()

A very common way to aggregate data in finance, is open-high-low-close.

rwalk_df.resample('W').ohlc().tail()

rwalk_df.resample('W').ohlc().plot(figsize=(10,10))
plt.show()

A very important topic in time series analysis is filtering with moving windows.

With rolling, we can easily create a moving average from a time series.

rwalk_df.plot(linewidth=0.5)
rwalk_df.rolling(25).mean().plot()
plt.show()

As you can see, by default the function calculates the average only when all values are awailable, and therefore, there are missing values at the beginning. You can avoid this with the **min_periods** parameter. With the **center** parameter, you can remove the lag in the average:

rwalk_df.plot(linewidth=0.5)
rwalk_df.rolling(25,min_periods=1,center=True).mean().plot()
plt.show()

Very often, the moving average is calculated using an exponentially weighted filter. Pandas has the **ewm** function for that.

rwalk_df.plot(linewidth=0.5)
rwalk_df.ewm(span = 25,min_periods=1).mean().plot(color='darkred',figsize=(10,10))
rwalk_df.rolling(25,min_periods=1).mean().plot(color='darkgreen')
plt.show()

With **rolling**, we can even calculate correlations from aggregates. Let's load a more interesting data for that.

euro_df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/EuStockMarkets.csv',index_col=0)

**freg='B'** means business day frequency. Here is the full list aliases for frequencies: [Offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)

euro_df.index = pd.date_range('1991-01-01', '1998-02-16', freq='B')

euro_df.plot()
plt.show()

The returns of stock indices.

euro_ret_df = euro_df.pct_change()

euro_ret_df.iloc[:100].plot(figsize=(10,5))
plt.show()

The correlation between DAX and the other indices calculated from a 64-day window.

euro_ret_df['DAX'].rolling(64).corr(euro_ret_df[['SMI','CAC','FTSE']]).plot(linewidth=1,figsize=(10,5))

With **apply**, you can even define your own functions to calculate an aggregate value from windows. The following calculates an interquartile range from the window.

inter_quart = lambda x: np.quantile(x,0.75)-np.quantile(x,0.25)

euro_ret_df.rolling(64).apply(inter_quart).plot(linewidth=1,figsize=(10,5))
plt.show()

Time series analysis in Pandas is a vast topic, and we have only scratched a surface. If you want to learn more, check [pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)

#### ARIMA models

In this section, I will shortly introduce ARIMA models. They assume that in a time series the present valua depends on the previous values (AR = autoregressive). They also include moving averages (MA). "I" in the name means that the modeled time series is the n:th difference of the original time series (I = integrated):

Let's build an ARIMA model for the euro dataset.

One essential tool to assess how the data depends on the previous values and select a model that reflects this is the sample autocorrelation function. The sample ACF will provide us with an estimate of the ACF and suggests a time series model suitable for representing the data's dependence. For example, a sample ACF close to zero for all nonzero lags suggests that an appropriate model for the data might be iid noise.

from pandas.plotting import autocorrelation_plot

Lags 1, 6, 11 and 13 appear to be statistically signficant.

autocorrelation_plot(euro_ret_df['FTSE'][1:],marker='o')
plt.xlim(0,20)
plt.ylim(-0.1,0.1)
plt.show()

from statsmodels.tsa.arima.model import ARIMA

Let's separate the last 100 observations from the data for testing.

train_set, test_set = euro_df['FTSE'][:-100].to_list(), euro_df['FTSE'][-100:].to_list()

Based on the ACF plot, 13 lags, integrated=1 (first difference of the original series) and first order moving average.

model = ARIMA(train_set, order=(13,1,1))
model_fit = model.fit()

model_fit.summary()

The residuals of the model (difference between the prediction and the true value).

plt.plot(model_fit.resid[1:50])
plt.show()

The distribution of the residuals.

sns.kdeplot(model_fit.resid[1:])
plt.show()

Prediction 100 steps forward and comparison to real values. As you can see from the plot below, the prediction is heavily based on the current value. The best prediction for tomorrow is the current value.

temp_set = train_set
predictions = []

for step in range(len(test_set)):
    model=ARIMA(temp_set, order=(5,1,1))
    results = model.fit()
    prediction = results.forecast()[0]
    predictions.append(prediction)
    temp_set.append(test_set[step])

plt.figure(figsize=(10,5))
plt.plot(predictions,color='red')
plt.plot(test_set)
plt.show()

A comprehensive list of model diagnostics can be plotted with **plot_diagnostics**.

model_fit.plot_diagnostics(figsize=(10,10))

