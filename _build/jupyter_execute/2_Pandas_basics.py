## Pandas data basics

We will mostly use Pandas dataframe as our data type. This and the next chapter will introduce the dataframes in detail. Pandas tries to pick up the best features of Numpy, spreadsheets and relational databases, and actually succeeds mostly in it. Pandas is an unbelievably powerful data manipulation tool.

Before we start exploring Pandas, it is good to understand how the standard data types in Python work. So, let's begin with them.

### Standard data types

#### List
A list is probably the most used data type in Python. It is a variable-length collection of items. A list can be defined in Python using either square brackets or the **list** function. Lists in Python are mutable. You can change the values of a list afterwards.

example_list = [1,2,'three',4]

example_list[2]

example_list[2] = 3

example_list

You can add a new element to the end of the list with **append()**.

example_list.append(5)

example_list

You can also add an item to a specific location in the list using **insert()**.

example_list.insert(3,3.5)

example_list

With **pop()** you can remove items from the list.

example_list.pop(3)

example_list

You can remove specific item from the list using **remove()**.

example_list.remove(3)

example_list

Lists are commonly used to acquire items from generators. Here is an example:

small_gen = (x**2 for x in range(10)) # a generator
list_of_gen = [item for item in small_gen]

list_of_gen

You can check if an item is or is not in the list with **in** and **is not**.

16 in list_of_gen

17 in list_of_gen

With **+** you can combine lists.

example_list + example_list

You can extend a list with multiple elements with **extend**.

example_list.extend([6,7,8])

example_list

You can use **sort()** to sort a list. It organises the original list. Pandas, in contrary, creates a new object in this kind of operations.

unsorted_list = [7,3,6,4,8,2]

unsorted_list.sort()

unsorted_list

Slicing is an important concept in Python. The sections of most sequency types can be selected with slice notation.

small_list = list(range(10))

small_list[1:3]

small_list[0:3]

small_list[:3]

small_list[3:]

small_list[-3:]

small_list[-5:-3]

**Step** can be used to define the frequency of slicing.

small_list[::2]

small_list[1:8:2]

small_list[::-1]

#### Tuple

A tuple is an immutable sequence of Python objects. Because it is immutable, it has also fixed length. You can create tuples using parantheses.

small_tuple = (10,20,30,40)

small_tuple

Otherwise, tuples are very similar to lists in Python. For example, you can access elements with square brackets.

small_tuple[2]

And you can concatenate them using **+**.

small_tuple + small_tuple

But, they are immutable. You can not change values in them.

small_tuple[2] = 35

Python will automatically unpack the tuple, if it is fed to a collection of variables (remember how a function returns a tuple).

a,b,c,d = small_tuple

a

b

Because a tuple is immutable, there are not many tuple methods. One useful is **count**, which counts the occurrences of a value.

small_tuple.count(40)

#### Dict

**Dict** is the most complicated built-in data type in Python. It is a mutable collection of key-value pairs of Python objects. Dicts are created using curly braces.

sample_dict = {'a': 500, 'b' : 1000, 'c' : 1500, 'd': 2000}

You can access and add elements of a dict using brackes with a key inside them.

sample_dict['a']

sample_dict['e'] = 2500

sample_dict

There is no restrction that the keys should be strings and the values should be numbers.

sample_dict[6] = 'some_text'

sample_dict

You can delete an item from a dict using **del**.

del sample_dict[6]

sample_dict

The **keys** and **values** methods can be used to get iterators of the dict's keys and values. (Remember that the list method was commonly used to extract the contents of an iterator.)

list(sample_dict.keys())

list(sample_dict.values())

You can merge two dictionaries with **update**.

sample_dict.update({'f':3000,'g':3500})

sample_dict

#### Set

A set in Python is a collection of unique elements. They can be created with **set** or with curly braces.

set([5,5,5,6,6,6,7,7,7,8,8,8])

{4,4,4,5,5,5,6,6,6,7,7,7}

A Python set is very similar to a mathematical set, so it has methods for mathematical set operations.

x = {3,4,5,6,7,8}
y = {6,7,8,9,10}

x.union(y)

x.intersection(y)

x.difference(y)

x.symmetric_difference(y)

#### Comprehensions

Comprehensions, especially the list ones, are a great feature in Python. With them, you can efficiently create collections with a single statement. Let's look some examples. (**%** is a module operator)

squares = [x**2 for x in range(10) if x%2==0]

squares

You could do exactly the same with a much longer for loop.

squares = []
for x in range(10):
    if x%2==0:
        squares.append(x**2)

squares

Here is an another example:

fd = open('quotes.txt',encoding='utf-8')

[text.upper().rstrip() for text in fd if len(text) < 50]

Python also has set and dict comprehensions that work similarly. Here is an example. First, we set our stream position to the beginning

fd.seek(0)

The following command creates a dict where the keys are the first five letters of each quote and the values are the lengths of the quotes (if the quote is less than 50 characters long).

{item[0:5] : len(item) for item in fd if len(item) < 50}

Nested list comprehensions are useful, but somewhat difficult to grasp. Let's look some examples.

fd.seek(0)

In the following, we split the quotes to a list of words, creating a list of lists

quotes = [quote.rstrip().split(" ") for quote in fd if len(quote) < 45]

quotes

Using a nested list comprehension, we can now collect individual words to a list.

[word for quote in quotes for word in quote]

Always remember to close the file.

fd.close()

### Pandas data structures

Two main data structures of Pandas are **Series** and **Dataframe**. Let's analys both of them.

#### Pandas series

Let's start by importing Pandas.

import pandas as pd

A pandas series is a one-dimensional array of Numpy-type objects, and an associated array of labels, called **index**.

first_pandas = pd.Series([5,3,7,3,7,1])

first_pandas

If we do not pass index when defining a series, it will be given a default index of sequential values.

first_pandas.index

You can define a custom index when creating a series.

second_pandas = pd.Series([1,2,3], index=['first','second','third'])

second_pandas

A pandas series works like a dict. You can use the index to pick up a single value.

second_pandas['second']

You can also use boolean indexing. Notice how the following command picks up the indices 0, 2 and 4.

first_pandas[first_pandas > 4]

You can use Numpy operations to your Pandas series.

import numpy as np

first_pandas*2

np.log(first_pandas)

As you have noticed, Pandas Series is very similar to a Python dict. Actually, you can very easily create a Pandas series from a dict.

sample_dict = {'a': 500, 'b' : 1000, 'c' : 1500, 'd': 2000}

sample_series = pd.Series(sample_dict)

sample_series

Important feature in Pandas is that everything is aligned by index.

sample_series2 = pd.Series([100,200,300,400], index=['c','a','d','b'])

sample_series + sample_series2

If you look carefully, you can see that summing was done "index-wise".

You can give a name to the values and index of your Series.

sample_series.name = 'some_values'
sample_series.index.name = 'some_letters'

sample_series

You can also change your index values very easily.

sample_series.index = ['aa','bb','cc','dd']

sample_series

#### Dataframe

A Pandas Dataframe is a rectangular table of objects with similar features as a Pandas Series. Each column can have different type of values. In a way, it is a 2-dimensional extension of Pandas Series. For example, it has both a row and a column index.

A dataframe can be constucted in many ways. One popular option is to use a dict of equal-length lists.

company_data = {'year': ['2015','2016','2017','2018','2019','2020'],
                'sales': [102000, 101000, 105000, 115000, 111000, 109000],
               'profit': [5000, 6000, 8000, 7000, 9000, 3000]}

company_df = pd.DataFrame(company_data)

company_df

You can define the order of columns when creating a dataframe.

pd.DataFrame(company_data, columns = ['year','profit','sales'])

If you add a column to the list that is not included in the dict, it will be added to the dataframe with missing values.

company_df = pd.DataFrame(company_data, columns = ['year','sales','profit','total_assets'])

company_df

You can pick up one column from a dataframe using the usual bracket-notation. It will return the column as a Pandas series.

company_df['profit']

Value assignment to a dataframe is easy. You can either add one value that is repeated in every instance of a column.

company_df['total_assets'] = 105000

company_df

Or you can add a full list of values.

company_df['total_assets'] = np.random.randint(105000,120000,6)

company_df

Remember that if you add a Pandas series to a datarame, the values will be matched according to their index values!  For missing index values, a NaN-value will be added.

debt_sr = pd.Series([9000,6000,8000,10000],index=[2,0,5,3])

debt_sr

company_df['debt']  = debt_sr

company_df

There are many ways to remove columns. One option is to use **del** that works also with other Python data types.

del company_df['debt']

company_df

A dict of dicts can also by used to create a dataframe. The keys of the outer dict are the columns and the keys of the inner dict are the index values. (Btw., In real applications we are not "writing" our data. Almost always we load the data from a file, using for example, Pandas **read_csv()**. So, do not worry.) 

company_dict = {'company a': {2010: 10.9, 2011: 8.7, 2012: 9.5},'company b': {2010: 11.2, 2011: 9.6, 2012: 8.8}}

new_df = pd.DataFrame(company_dict)

new_df

Transposing data is done in the same way as in Numpy, just by adding **.T** after the name of the dataframe.

new_df.T

There are still many other ways to construct a dataframe, and we will use some of then in the following chapters. A detailed introduction to constructing a dataframe can be found here: [pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html)

There is actually a third data type in Pandas, **Index objects**. They are used to hold the labels of columns and index of a dataframe or series. Pandas **index** is immutable and has similiraties with Python sets. However, Pandas **index** can contain duplicate items.

new_df.index

### Basic Pandas functions

Let's use a more realistic example for the following examples.( More about **read_csv()** in the later chapters.)

big_companies_df = pd.read_csv('big_companies.csv',index_col='NAME',delimiter=';')

big_companies_df

With **reindex()**, you can rearrange the values of the dataframe. Let's load Numpy.

import numpy as np

The following command shuffles the order of indices (company names).

random_order = np.random.permutation(big_companies_df.index)

random_order

big_companies_df.reindex(index = random_order)

You can also rearrange columns with **reindex()**.

random_columns = np.random.permutation(big_companies_df.columns)

random_columns

big_companies_df.reindex(columns = random_columns)

If have names in the reindex list that are not included in the original index/columns, these are added to the dataframe with missing values. You can prevent those missing values, for example, with **ffill** in the parameter **method** (there is also **bfill** and **nearest**).

new_df = big_companies_df.reindex(['APPLE','MICROSOFT','TESLA'])

new_df

**Ffill** is a forward fill. With strings, it adds the values of the original dataframe forward, according to the alphabetical order of the strings. Because **APPLE** is after **AMAZON.COM** and **ALIBABA...** in alphabetical order, NaN values are put to these new rows.

new_df.reindex(big_companies_df.index,method = 'ffill')

It is easy to drop values by indices from a series/dataframe using **drop()**. By default, Pandas drops indices. Defining **axis=1**, you can drop columns. Another option is to drop both using **index** and **columns**

big_companies_df.drop('APPLE')

big_companies_df.drop('DIVIDEND YIELD',axis=1)

big_companies_df.drop(index=['APPLE','TESLA'],columns=['DIVIDEND YIELD','P/E - RATIO'])

An important thing to notice is that most of the Pandas methods are non-destructive. In the previous operations, we have made many manipulations to the big_companies_df dataframe. However, it is still the same.

big_companies_df

Pandas makes almost all manipulations so that the manipulated dataframe is returned as a new object. If you want to change the original dataframe, almost all the methods have **inplace=True** for that.

Indexing, selecting values and slicing Pandas dataframes is very similar to other Python objects. One important difference is that when using labels for slicing, the last label is inclusive.

big_companies_df['MICROSOFT':'TESLA']

You can also pick up columns using double brackets, put the slicing is not possible.

big_companies_df[['DIVIDEND YIELD','CASH FLOW/SALES']]

You can also pick up values using booleans.

big_companies_df[big_companies_df['P/E - RATIO'] > 40]

You can build a boolean dataframe easily. In the following, we check if a value in the dataframe is larger than 10.

big_companies_df > 10

To make the indexing more clear, Pandas has operators **loc** and **iloc** that can be used to select the rows and columns from a dataframe using Numpy-like notation. **loc** works with labels and **iloc** with integer positions.

big_companies_df

Notice that with **loc** we can use slicing also with columns.

big_companies_df.loc['TENCENT HOLDINGS':'TAIWAN SEMICON.MNFG.','DIVIDEND YIELD':'P/E - RATIO']

big_companies_df.iloc[6:12,0:5]

When adding dataframes, any index-column pairs that are missing from either dataframe, will be replaced with NaN-values. Let's look an example. We create two dataframes with partial data. Both are missing one (different) row and one column.

partial_data_df = big_companies_df.drop(index='APPLE',columns='DIVIDEND YIELD')

partial_data_df

partial_data2_df = big_companies_df.drop(index='TESLA',columns='CASH FLOW/SALES')

partial_data2_df

Those cells that are found in both dataframes have their values summed. If a specific cell is only in one of the dataframes, a NaN value is inserted to that location in the resulting dataframe.

partial_data_df + partial_data2_df

If you want to have some other than NaN values to these locations, you can use the **add()** method with the **fill_value** parameter. This will  replace the missing values with a specified value. Notice how **APPLE** **CASH FLOW/SALES** and **TESLA**/**DIVIDEND YIELD** still have missing values. This is because they are missing from **both** partial dataframes.

partial_data_df.add(partial_data2_df,fill_value=0)

You can also do arithmetic operations between Dataframe and Series. One thing to note is that these operations use **broadcasting** like Numpy arrays. Broadcasting is a very important conceopt in ML. You can read more about it here. [numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html)

In its simplest, broadcasting means that if we are subtracting a Series from a Dataframe, the Series is subtracted from every row (or column) of the dataframe.

In the following, we subtract the values of **TESLA** from every row of our dataframe.

sample_series = big_companies_df.loc['TESLA']

sample_series

The index of the series is matched with the columns of the dataframe.

big_companies_df-sample_series

If the index/columns do not match, the resulting dataframe is a union.

sample_series = sample_series.append(pd.Series(1,index=['extra row']))

big_companies_df-sample_series

Basically all the Numpy's element-wise array methods work with Pandas objects.

Tesla looks a little bit better now (no negative returns). :)

np.abs(big_companies_df)

If you want to apply a function to rows or columns of a dataframe, you can use **apply**. 

differ = lambda x:np.max(x)-np.min(x)

**differ** calculates the difference between the maximum and minimum of each column.

big_companies_df.apply(differ)

We can also do this row-wise.

big_companies_df.apply(differ,axis=1)

For element-wise manipulations, you can use **applymap()**.

big_companies_df.applymap(lambda x: -x)

**sort_index** can be used for sorting.

big_companies_df.sort_index()

big_companies_df.sort_index(axis=1)

If you want sort by values, you can use **sort_values()**

big_companies_df.sort_values(by = 'P/E - RATIO',ascending=False)

You can use many colums when sortin by values.

big_companies_df.sort_values(by = ['DIVIDEND YIELD','P/E - RATIO'],ascending=[True,False])