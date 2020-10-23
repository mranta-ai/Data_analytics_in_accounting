## Connecting with accounting databases

### Power BI

The easiest way to exploit the Python ecosystem in Power BI is to enable Python scripting inside Power BI. For that, you need to have Python installed in your system.

In Power BI, check from **Options - Global - Python scripting** that you have correct folders for your Python and IDE. Power BI should detect the folders automatically, but it is good to check.

Furthermore, you need to have at least Pandas, Matplotlib and Numpy installed.

You can run Python scripts inside Power BI using **Get Data - Other - Python script**. However, it is a good habit to check in your Python environment that the script is working.

There are a few limitations with the connection between Power BI/Python:
- If you want to import data, it should be represented in a Pandas data frame.
- The maximum run time of a script is 30 minutes.
- You must use full directory paths in your code, not relative paths.
- Nested tables are not supported.

Otherwise, implementing Python in Power BI is very similar to doing analysis purely inside Python. The good side is, of course, that you have the tools of both Python and Power BI at your disposal.

### MySQL, SAP and others

There are good libraries for connecting to MySQL, for example, MySQLdb: [pypi.org/project/MySQL-python/](https://pypi.org/project/MySQL-python/). If you want, you can use your MySQL database to go through the examples, instead of SQlite.

SAP HANA is used to connect with Python to a SAP database. Here are the instructions on how to connect Python to SAP: [developers.sap.com/tutorials/hana-clients-python.html](https://developers.sap.com/tutorials/hana-clients-python.html). 
The task is quite difficult, and we are not doing that in this course.

### Sqlite

In the following, we will analyse our example database purely in Python. For that, we use an example company database that is available here: [github.com/jpwhite3/northwind-SQLite3](https://github.com/jpwhite3/northwind-SQLite3)

![Northwind](./images/northwind.png)

This demo-database has originally been published to Microsoft Access 2000. We analyse it using Sqlite3. However, keep in mind that you can repeat the following analysis by connecting to many other databases, like SAP. In the following, we use some basic SQL statements. However, this course is not about SQL, so we do not go deeper in that direction.

Sqlite3 is included in the standard library. So, you do not need to install any additional libraries. Let's start by importing the library.

import sqlite3

We can create a connection to the example database with **connect()**. You need to have the **Northwind_large.sqlite** file in your work folder for the following command to work.

connection = sqlite3.connect('Northwind_large.sqlite')

**cursor()** returns a cursor for the connection. A cursor -object has many useful methods to execute SQL statements.

cursor = connection.cursor()

The tables of the database can be collected with the following commands. **execute()** is used for a SQL statement and **fetchall()** collects all the rows from the result.

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())

We check the fields of the **Employees**, **OrderDetails**, **Orders** and **Products** tables. The outpus are messy lists of tuples. The name of a field is always the second item in a tuple.

cursor.execute("PRAGMA table_info(Employees)")
print(cursor.fetchall())

cursor.execute("PRAGMA table_info(OrderDetails)")
print(cursor.fetchall())

cursor.execute("PRAGMA table_info(Orders)")
print(cursor.fetchall())

cursor.execute("PRAGMA table_info(Products)")
print(cursor.fetchall())

Pandas has a very convenient function, **read_sql_query**, to load SQL queries to dataframes. Let's start by loading Pandas.

import pandas as pd

SQL queries are a whole new world, and we use only the essential. The following code picks up **LastName** from the **Employees** table, **UnitPrice** and **Quantity** from the **OrderDetails**, **OrderDate** and **ShipCountry** from **Orders**, **CategoryName** from **Categories**, and **ProductName** from **Products**. The next part of the code is important. The **JOIN** commands connect the data in different tables in a correct way. Notice how we qive our sqlite3 database connection -object as a paramter to the function.

query_df = pd.read_sql_query("""SELECT Employees.LastName, OrderDetails.UnitPrice, 
OrderDetails.Quantity, Orders.OrderDate, Orders.ShipCountry, Categories.CategoryName, Products.ProductName
FROM OrderDetails
JOIN Orders ON Orders.Id=OrderDetails.OrderID
JOIN Employees ON Employees.Id=Orders.EmployeeId
JOIN Products ON Products.ID=OrderDetails.ProductId
JOIN Categories ON Categories.ID=Products.CategoryID""", connection)

Now that we have everything neatly in a Pandas dataframe, we can do many kinds of analyses. The other chapters focus more on the Pandas functionality, but let's try something that we can do.

For example, to analyse trends, we can change **OrderDate** to a datetime object with **to_datetime()**.

query_df['OrderDate'] = pd.to_datetime(query_df['OrderDate'])

query_df

Next, we can change our datetime object as index. 

query_df.index = query_df['OrderDate']

query_df

We still need to order the index.

query_df.sort_index(inplace=True)

Let's calculate the total number of orders for different product categories.

query_df['CategoryName'].value_counts().plot.bar()

Let's check next, to which country the company is selling the most. The following command is quite long! First, it groups values by **ShipCountry**, then counts values and sorts them in a descending order by **Quantity**, and finally selects only **Quantity** -column.

query_df.groupby('ShipCountry').count().sort_values('Quantity',ascending=False)['Quantity']

A nice thing in Python (and Pandas) is that we change the previous to a bar chart just by adding **plot.bar()** to the end of the command.

Let's first load Matplotlib to make our plots prettier.

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

query_df.groupby('ShipCountry').count().sort_values('Quantity',ascending=False)['Quantity'].plot.bar(figsize=(10,5))
plt.show()

With pivot tables, we can do a 2-way grouping.

query_df.pivot_table(index='LastName',columns='ShipCountry',values='Quantity',aggfunc=sum)

With **Quantity** and **OrderPrice**, we can calculate the total price of the orders. Using * for multiplication, Python/Pandas makes the multiplication element-wise.

query_df['TotalPrice'] = query_df['Quantity']*query_df['UnitPrice']

There is too much information to one plot, so let's resample the data before plotting (**'M'** means monthly).

query_df['TotalPrice'].resample('M').sum().plot(figsize=(10,5),style='r--')

Let's plot how the sales of different salesperson have progressed. The **for** loop in the command goes through all the salesperson and draws their performance to the same chart. With **set**, we can pick from the **LastName** column unique values.

plt.figure(figsize=(15,10))
for name in set(query_df['LastName']):
    plt.plot(query_df['TotalPrice'][query_df['LastName'] == name].resample('Q').sum())
plt.legend(set(query_df['LastName']))
plt.show()

We can also use bar charts. Here are the sales of different salesperson and product categories. We first do a two-level grouping, sum the values in those groups, and pick **TotalPrice**. Adding **plot.barh()** to the end turns the 2-level grouping table into a bar chart. 

query_df.groupby(['LastName','CategoryName']).sum()['TotalPrice'].plot.barh(figsize=(5,15))

We can also use percentage values in tables. (I admit, the following command is a mess!). It divides the values of a **LastName/CategoryName** -pivot table with the row sums of that table. Then, it multiplies these numbers by hundred. **style.format** is used to decrease the number of decimals to 2 **2f**, and to add **%** to the end.

(query_df.pivot_table(values = 'TotalPrice', index = 'LastName',
                      columns = 'CategoryName').divide(query_df.pivot_table(values = 'TotalPrice',
                    index = 'LastName', columns = 'CategoryName').sum(axis=1),axis=0)*100).style.format('{:.2f} %')

### Other sources of accounting data

Google Dataset Search is an excellent source of datasets, including interesting accounting datasets: [datasetsearch.research.google.com/](https://datasetsearch.research.google.com/)

Quandl is another interesting source of data. They have some free datasets, but you need to register to get an api key before you can download any data. Quandl offers a library that you can use to download datasets directly in Python: [www.quandl.com/](https://www.quandl.com/)

#### Pandas Datareader
Pandas Datareader is a library that can be used to download external datasets to Pandas dataframes. [pandas-datareader.readthedocs.io/en/latest/](https://pandas-datareader.readthedocs.io/en/latest/)

Currently, the following sources are supported in Pandas Datareader
* Tiingo
* IEX
* Alpha Vantage
* Enigma
* EconDB
* Quandl
* St.Louis FED (FRED)
* Kenneth French’s data library
* World Bank
* OECD
* Eurostat
* Thrift Savings Plan
* Nasdaq Trader symbol definitions
* Stooq
* MOEX
* Naver Finance

For most of these, free registration is required to get an API key.

You need to install Datareader first. It is included in Anaconda and can also be installed with Pip using a command **pip install pandas-datareader**.

Let's import the library

import pandas_datareader as pdr

Let's use in our example EconDB ([www.econdb.com/](https://www.econdb.com/)). In the following code, we load the quarterly values of Finland's gross domectic product from the year 1999 to the most recent value.

data = pdr.data.DataReader('ticker=RGDPFI','econdb',start=1999)

It returns a Pandas dataframe, to which we can apply all the Pandas functions.

data.plot(figsize=(10,5))
plt.show()

Fama/French data library is also very interesting for accounting research. [mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

import pandas_datareader.famafrench as fama

There are 297 datasets available.

len(fama.get_available_datasets())

Let's load an industry portfolio return data.

data2 = pdr.data.DataReader('10_Industry_Portfolios', 'famafrench')

This time, it returns a dictionary. The items of the dictionary are dataframes with different data. **DESCR** can be used to get information about the data.

type(data2)

data2['DESCR']

data2[0].plot(figsize=(10,8))

