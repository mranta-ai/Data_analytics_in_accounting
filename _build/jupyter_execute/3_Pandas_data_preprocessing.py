## Pandas Data preprocessing

You might be wondering why we are talking so much about data processing and manipulation in this book. "Where are all the fancy ML methods" you might  ask. Unfortunately, most of the time in data science goes to data pre-processing. It is often reported that 80 % of a data scientist's time goes to data preparation: cleaning, transforming, rearranging and creating suitable features (feature engineering). So, to be a succesfull data scientist, you need know how to play with data. Luckily, we have Pandas at our disposal, wich is one of the most powerful data manipulation tools available.

Let's look at some of the most common preprocessing situatsions we encounter when doing data science.

### Basic methods

Missing data is probably the most common issue with data.

import pandas as pd

companies_df = pd.read_csv('emissions.csv',index_col='NAME',delimiter=';')

companies_df

**Dropna** can be used to filter out missing data. The **how** parameter defines do all/any of the values in rows/columns need to be zero for dropping.

companies_df.dropna(how='all')

companies_df.dropna(how='any')

You can again use the **axis** parameter. However, it is not meaningful to use here because every column has NaN-values (**any** returns an empty table), and not all values are NaN in any column (**all** returns the original dataframe).

With **thresh** you can define how many of the values are allowed to be NaN.

companies_df.dropna(thresh=2)

You can use **fillna** to fill NaN values with other values.

companies_df.fillna(0)

Using a dict, you can define different fill values for different columns. A very common choice is to use the mean value of each column.

companies_df.fillna({'Board member compensation':1000000, 'Total CO2 To Revenues':10,
       'Emission Reduction Target %':0})

Interpolation methods that were available for reindexing can be used with fillna.

companies_df.fillna(method = 'bfill')

**duplicated()** can be used to drop duplicated rows. It drops only duplicates that are next to each other

More about **append()** later. Here we just build a new dataframe with companies_df in it twice. And then we sort the index so that every company is twice in the new dataframe.

new_df = companies_df.append(companies_df)

new_df.sort_index(inplace=True)

new_df

new_df.duplicated()

We can remove duplicated rows using **drop_duplicated()**.

new_df.drop_duplicates()

It is easy to apply function transformation to rows, columns or individual cells of a dataframe. **map** -metohd can be used for that.

companies_df['Compensation ($ millions)'] = companies_df['Board member compensation'].map(
    lambda x: "{:.2f}".format(x/1000000)+" M$")

companies_df

**Map()** can also be used to transform index. The following command turns the company names backwards.

companies_df.index.map(lambda x : x[::-1])

Of course, you can also use **rename()**. Using a dictionary, **rename()** can also be used to change only some of the labels.

companies_df.rename(index=str.title)

**Replace()** can be used to replace any values, not just NaN values. You can pass also dict/list, if you want to replace multiple values.

import numpy as np

companies_df.replace(np.nan,-999) # Pandas NaN is np.nan

Dividing data to **bins** is a very important feature in Pandas. You can use **pd.cut()** (notice that it is not a dataframe method) to creata a categorical object. **pd.value_counts()** can be used to calculate the number of observations in each bin. With the **labels** parameter you can define names for different bins.

compensation_bins = [10000,100000,1000000,10000000,100000000]

comp_categ = pd.cut(companies_df['Board member compensation'],compensation_bins)

comp_categ

comp_categ.values.categories

comp_categ.values.codes

pd.value_counts(comp_categ)

pd.cut(companies_df['Board member compensation'],
                    compensation_bins,labels = ['Poor board members','Rich board members',
                                                'Very rich board members','Insanely rich board members'])

If you pass a number to the **bins** parameter, it will return that many equal-length bins.

pd.cut(companies_df['Total CO2 To Revenues'],4,precision=1)

For many purposes, **qcut()** is more useful as it bins the data based on sample quantiles. Therefore, every bins has approximately the same number of observations. You can also pass specific quantiles as a list to the function.

pd.qcut(companies_df['Total CO2 To Revenues'],4)

Detecting and filtering outliers is easy with boolean dataframes

companies_df[companies_df['Board member compensation'] < 10000000]

Filtering is also possible with the summary statistics of variables. The following command picks up cases where the values of **all** variables deviate from the mean less than two standard deviations.

companies_df.drop('Compensation ($ millions)',axis=1,inplace=True)

companies_df[(np.abs(companies_df-companies_df.mean()) < 2*companies_df.std(axis=0)).all(1)]

Winsorising is another common procedure in practical econometrics. In that method, the most extreme values are moved to specific quantiles, usually 1% and 99% quantiles. It is easiest to implement with the **clip()** method. Notice how the following command winsorise all the variables.

companies_df.clip(lower = companies_df.quantile(0.1), upper = companies_df.quantile(0.9),axis=1)

Random sampling from data is easy. You can use the **sample()** method for that. It is also possible to sample with replacement that is needed in many numerical statistical methods.

companies_df.sample(n=10,replace=True)

If you want to randomise the order of values in a dataframe, you can use Numpy's **random.permutation()**

permut_index = np.random.permutation(len(companies_df))

companies_df.iloc[permut_index]

Very often in econometrics, you need to transform your categorical variables to a collection of dummy variables. It easily done in Pandas using the **get_dummies()** function.

pd.get_dummies(companies_df['Emission Reduction Target %'],prefix='Emiss_')

Pandas has efficient methods to manipulate strings in a dataframe. Due to this, Pandas is very popular among researchers that need to use string data in their analysis. This makes it also a very important tool for accounting data analysis. All the Python string object's built-in methods can be used to manipulate strings in a dataframe. We already discussed string-methods in Chapter 1.

Regular expressions can also be used to manipulate string-dataframes. Regular expressions is a very broad topic, and it takes time to master it. Let's look at some very simple examples. If we want to split a sentence to a words-list, repeated whitespaces make the process difficult with the standard string-methods. However, it is very easy with regular expressions.

import re

The regex for multiple whitespaces is **\s+**.

splitter = re.compile('\s+')
splitter.split("This    is   a    sample        text.")

Regular experssions is a powerful tool, but very complex. You can search email-addresses from a text with the following regex-command.

reg_pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

email_search = re.compile(reg_pattern, flags = re.IGNORECASE)

example = """The general format of an email address is local-part@domain,
and a specific example is jsmith@example.com. An address consists of two parts. 
The part before the @ symbol (local part) identifies the name of a mailbox. 
This is often the username of the recipient, e.g., jsmith. 
The part after the @ symbol (domain) is a domain name that represents 
the administrative realm for the mail box, e.g., a company's domain name, example.com."""

email_search.findall(example)

We do not go into regular expression more at this point. But we will see some applications of them in the following chapters. If you want to learn more about regular expressions, here is a good website: [www.regular-expressions.info/index.html](https://www.regular-expressions.info/index.html)

If your string data contains missing values, usually the standard string methods will not work. Then you need to use the dataframe string methods.

You can slice strings normally.

companies_df.index.str[0:5]

And you can use regular experssions, for example, to search strings.

companies_df.index.str.findall('am', flags = re.IGNORECASE)

Here is a good introduction to Pandas string methods. [pandas.pydata.org/pandas-docs/stable/user_guide/text.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html). If you scroll down, there is a full list methods.

### Advanced methods

Let's look some more advanced methods of Pandas next. We need a new dataset for that.

electricity_df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/USAirlines.csv',index_col=0)

electricity_df['firm'].replace({1:'A',2:'B',3:'C',4:'D',5:'E',6:'F'},inplace=True)

The cost function of electricity producers.

electricity_df

There are many tools to deal with data that has more than two dimensions. The basic idea is to use hierarchical indexing in a two dimensional dataframe. For example, we can index the data by firm-years using **set_index()**.

electricity_df.set_index(['firm','year'],inplace=True)

electricity_df

Now the values are sorted so that all the years of a certain company are in adjacent rows. With **sort_index** you can order the values according to years.

electricity_df.sort_index(level=1)

Two-level indexing enables an easy way to pick subgroups from the data. For a series, you can just use the standard indexing style of Python.

electricity_df['output']['A']

electricity_df['output'][:,1975]

With multi-index dataframes, you have to be a little bit more careful, because the subgroups are dataframes themselves. For example, you need to use **loc** to pick up the subgroups.

electricity_df.loc['A']

electricity_df.loc['B'].loc[1970]

If you want to pick values of a certain year, you change the order of indices using **swaplevel**.

electricity_df.swaplevel('firm','year').loc[1975]

You can easily calculate descriptive statistics at multiple levels. Most of the stat functions in Pandas include a **level** parameter for that. 

electricity_df.sum()

electricity_df.sum(level=0)

electricity_df.sum(level=1)

If we want to remove multi-index, we can use **reset_index()**. With the **level** parameter, you can decide how many levels from the multi-index are removed.

electricity_df.reset_index(level=1)

electricity_df.reset_index(inplace=True)

Summary statistics can also be calculated using the **groupby** method.

electricity_df.groupby('firm').mean()

electricity_df.groupby(['firm','year']).mean() # The mean of single values

**stack** and **unstack** can be used to reshape hierarchical index dataframes.

electricity_df.reset_index(inplace=True)

electricity_df

**Stack** turns a dataframe into a series.

data_series= electricity_df.stack()

data_series

**unstack()** can be used to rearragne data back to a dataframe.

data_series.unstack()

Yet another tool to reorganise data is **pivot_table**. More of it later.

### Merging datasets

Merging dataframes is often difficult for Pandas beginners. It can be a hassle. The usual cause of difficulties is to forget the importance of index with Pandas datatypes. Merging dataframes is not about gluing tables together. The merging is done according to indices.

gasoline_df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Gasoline.csv',index_col=0)

gasoline_df.set_index('country',inplace=True)

gasoline_df

Let's split the data.

gaso2_df = gasoline_df[['lrpmg','lcarpcap']]

gasoline_df.drop(['lrpmg','lcarpcap'],axis=1,inplace=True)

gasoline_df

gaso2_df

**Merge** combindes according to the index values

pd.merge(gasoline_df,gaso2_df,on='country')

You could also do this by *activating* both indices for merging.

pd.merge(gasoline_df,gaso2_df,left_index=True,right_index=True)

You can also use **join**. It uses the index values for merging by default.

gasoline_df.join(gaso2_df)

**Concat** is a more general tool for merging data. It is easiest to understand how it works, if we use very simple datasets. Let's create three simple Pandas Series.

x1 = pd.Series([20,30,40], index=['a','b','c'])
x2 = pd.Series([50,60], index=['e','f'])
x3 = pd.Series([70,80,90], index=['h','i','j'])

If we just feed them to **concat**, they are joined together as a longer Series.

pd.concat([x1,x2,x3])

If you change the axis, the result will be a dataframe, because there are no overlapping index values.

pd.concat([x1,x2,x3],axis=1)

x4 = pd.concat([x1,x2])

pd.concat([x4,x2],axis=1)

If you just want to keep the intersction of the series, you can use **join='inner'**

pd.concat([x4,x2],axis=1,join='inner')

### Data aggreagation

Data aggregation is one the most important steps in data preprocessing. Pandas has many tools for that.

**Groupby** is an important tool in aggragation. Let's load a new dataset for this.

import pandas as pd

comp_returns_df = pd.read_csv('comp_returns.csv',index_col=0)

comp_returns_df

Now, we can group data by the accounting standards variable and calculate the mean. Notice how Pandas automatically drops the country variable, because you cannot calculate a mean from strings:

comp_returns_df.groupby('ACCOUNTING STANDARDS FOLLOWED').mean()

You can make a two-level grouping by adding a list of columns to **groupby**.

comp_returns_df.groupby(['ISO COUNTRY CODE','ACCOUNTING STANDARDS FOLLOWED']).mean()

There are many other groupby-methods, like **size()**. The full list of **groupby** -methods are here: [pandas.pydata.org/pandas-docs/stable/reference/groupby.html](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html)

comp_returns_df.groupby('ISO COUNTRY CODE').size()

If we just use the **groupby** method without a following method, it will return a Pandas groupby object.

comp_returns_df.groupby('ISO COUNTRY CODE')

This object can be used for iteration.

for name,group in comp_returns_df.groupby('ISO COUNTRY CODE'):
    print(name)
    print(len(group))

You could also group data according to columns using **axis=1**. It makes not much sense with this data, so we skip that example.

You can easily pick up just one column from a groupby object.

comp_returns_df.groupby('ISO COUNTRY CODE')['RETURN ON EQUITY - TOTAL (%)'].mean()

You could group data by defining a dict, a function etc. We do not go to these advanced methods here.

#### Data aggregation

Now that we know the basics of **groupby**, we can analyse more how it can be used to aggregate data. The methods of **groupby** are **count, sum, mean, median, std, var, min, max, prod, first and last**. These are opitmised for groupby objects, but many other methods work too. Actually, you can define your own functions with **groupby.agg()**.

def mean_median(arr):
    return arr.mean()-arr.median()

comp_returns_df.groupby('ACCOUNTING STANDARDS FOLLOWED').agg(mean_median)

Basically, those Pandas methods will work with **groupby** that are some kind of aggregations. For example, quantile can be used

comp_returns_df.groupby('ACCOUNTING STANDARDS FOLLOWED').quantile(0.9)

Even though **describe** is not an aggregating function, it will also work.

comp_returns_df.groupby('ACCOUNTING STANDARDS FOLLOWED').describe()

You can make the previous table more readable by unstacking it.

comp_returns_df.groupby('ACCOUNTING STANDARDS FOLLOWED').describe().unstack()

**Apply()** is the most versatile method to use with groupby objects.

def medalists(df,var = 'RETURN ON EQUITY - TOTAL (%)'):
    return df.sort_values(by=var)[-3:]

comp_returns_df.groupby('ACCOUNTING STANDARDS FOLLOWED').apply(medalists)

It  is also possible to use the bins created with **cut** or **qcut** as a grouping criteria in **groupby**.

ROE_quartiles = pd.qcut(comp_returns_df['RETURN ON EQUITY - TOTAL (%)'],4)

comp_returns_df.groupby(ROE_quartiles).size()

comp_returns_df.groupby(ROE_quartiles).mean()

Pivot tables are another option to organise data with Pandas. They are very popular in spreadsheet softwares, like Excel.It is very similar to cross-tabs that we use in statistics. Pandas has a specific function for pivot tables.

Let's load a new data.

large_df = pd.read_csv('large_table.csv',delimiter=';')

large_df.drop('Type',inplace=True,axis=1)

large_df

The default **pivot_table** aggregation type is mean.

large_df.columns

large_df.pivot_table(values='RETURN ON EQUITY - TOTAL (%)',
                     index='CSR Sustainability External Audit',columns='ACCOUNTING STANDARDS FOLLOWED')

You can use hierarchical index and add marginal sums to the table.

large_df.pivot_table(values='RETURN ON EQUITY - TOTAL (%)',
                     index=['CSR Sustainability External Audit','Audit Committee'],
                     columns='ACCOUNTING STANDARDS FOLLOWED',margins=True)

You can also use hierarchical columns

large_df.pivot_table(values='RESEARCH & DEVELOPMENT/SALES -',
                     columns=['CSR Sustainability Committee','Audit Committee'],
                     index='ISO COUNTRY CODE',margins=True)

Cross-tabs are a special case of pivot tables, where the values are frequencies.

large_df.pivot_table(columns='CSR Sustainability Committee',index='ISO COUNTRY CODE',
                     aggfunc='count',margins=True)

As you can see, it will return values for all variables. Therefore it is more convenient to use **crosstab**.

pd.crosstab(large_df['CSR Sustainability External Audit'],large_df['ISO COUNTRY CODE'],margins=True)