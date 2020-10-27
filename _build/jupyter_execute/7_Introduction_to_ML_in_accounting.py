## Introduction to Machine learning in accounting

In this chapter we will discus **shallow** machine learning models. Although neural networks can also be shallow, the discussion of them is postponed to Chapter 9.

### Ensemble methods

There is a saying: Two heads are better than one. What about even more heads? At least with machine learning, more heads is helpful (I am not sure about humans. :))

The idea of ensemble methods is to join many weak estimators as one efficient estimator. Using this approach, these methods achieve strong results. It is enough that the weak estimator is only slightly better than pure chance, their ensemble can still be a very efficient machine learning method.

**Example:** Let's assume that we have a weak estimator that can correctly predict the bankcruptcy of a company 52 % of a time. Thus, the predictor is only slightly better than pure chance (50 %).

However, an ensemble consisting of 100 weak estimators is correct 69,2 % of the time and an ensemble consisting of 1000 weak estimators is correct 90,3 %. of the time!

import scipy.stats as ss

binom_rv = ss.binom(100,0.52)
sum([binom_rv.pmf(i) for i in range(50,101)])

binom_rv = ss.binom(1000,0.52)
sum([binom_rv.pmf(i) for i in range(500,1001)])

In the following figure, one ellipse (a weak estimator) would a very bad classifier due to its incompatible shape with the two classes (the dots and diamonds). However, their ensemble is able to classify observations very well.

![Boost_ellips](./images/boost_ellips.png)

There are many options how the aggregate is calculated. It can be weighted average, majority, etc. depending on the application.

Very often the simple estimator in ensemble methods is the decision tree. In a decision tree, with conclusion based on the features of the model a tree structure is concstructed. From the leaves of the tree a prediction for the correct value/class can be inferred. The original decision tree structure gave only predictions for the correct class. The classification and regression trees (CART) have points instead classes in the leaves. This allows much more versatile interpretation and allows regression trees to be used also in regression applications.

Below is an example how decision trees are constructed. We have two features, equity ratio (ER) and return on assets (ROA). Based on these features, the companies are divided into three groups. First they are divided to two groups (ROA over or under *r*). Then companies in the (ROA < *r*) -group are divided based on the equity ratio (over or under *p*).

![dec_tree](./images/dec_tree.png)

The interpretation of symbols: diamond: no bankcruptcy risk, cross: low bankcruptcy risk, circle: high banckruptcy risk

![dec_tree](./images/dec_tree2.png)

The most common ensemble methods are bagging, random forest and boosting. They differ in how they decrease the correlation between their predictions. The benefit of ensemble methods increases when the correlation decreases.

The bagging (bootstrap aggregating) method decreases the correlation by feeding bootstrap samples to the weak estimators.

![bagging](./images/bagging.png)

The original random forest algorithm decreased correlation by feeding a subsample of features to the weak estimators (random subspace method). Later, the bootstrap aggregating of bagging was added to the method.

![random_forest](./images/rand_forest.png)

In recent year, boosting and especially gradient boosting has been a very popular ensemble method in applications.
In Boosting, weak estimators work in series. The idea is to feed the data again to a new weak learner so that the weight of misclassified points is increased. After training, the aggreaget estimate of the weak learners is calculated as a weighted mean. The largest weight is given to those learners, whose error function value was smallest.

![boosting](./images/boost.png)



Xgboost has probably been the most succesfull boosting method. It is very often behind the winning solutions of different machine learning competitions ([www.kaggle.com](https://www.kaggle.com)). Here is a short info from the Xgboost github-page: "XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, MPI, Dask) and can solve problems beyond billions of examples."

Later in the book, we will see an example using Xgboost.

Epälineaarinen vaikutuksen analysointi
Bootstrapilla mukaan myös tilastollinen merkitsevyys
Automaattinen mallin etsintä. Analyytikon ei tarvitse keksiä, onko riippuvuus muotoa 𝑥, 𝑥^2,𝑥^3 jne…


### Support vector machines
The Support Vector Machine (SVM) was previously one of the most popular algorithms in modern machine learning. It often provides very impressive classification performance on reasonably sized datasets. However, SVMs have difficulties with large datasets since the computations don’t scale well with the number of training examples. This poor performance with large datasets hinders somewhat their success in big data and is the reason why neural networks have partly replaced SVMs in that field. However, in accounting we have often datasets of modest size and SVMs work very well with them.

![SVM](./images/SVM_margin.png)

### Key ML libraries in Python

#### Numpy

Although Numpy is not exactly a machine learning library, it is the backbone of many other ML libraries and the most important library for numerical computing in Python. Therefore, we start our journey of ML libraries with the basics of NumPy.

The key feature of Numpy is its flexible and fast multidimensional **ndarray** that can contain large datasets. It enables mathematical operations between arrays in a way that is very similar to calculations with scalars.

import numpy as np

random_values = np.random.randint(100,200,(4,4))

random_values

Mathematical operations are then very easy to perform. The default is almost always element-vise operations.

random_values*10

random_values/100

random_values + random_values

np.log(random_values)

Every Numpy array has a shape parameter that can be used to check the shape of your array.

random_values.shape

Notice that the array can be, and very commonly is in machine learnig, more than two-dimensional. Here is a four-dimensional array.

randoms = np.random.randint(10,20,(2,2,2,2))

randoms

Numpy has **array()** for creating Numpy arrays. Many kinds of collections are accepted as inputs.

sample_list = [i**2 for i in range(10)]

sample_list

sample_np = np.array(sample_list)

sample_np

Two-dimensional arrays can be built from list of lists, etc.

sample_list2 = [[i,i**2] for i in range(10)]

sample2_np = np.array(sample_list2)

sample2_np.shape

You can check the dimensions with **ndim**.

sample2_np.ndim

You can quickly create arrays of zeros and ones with **zeros()** and **ones()**.

np.zeros((2,3))

np.ones((3,2))

Python is flexible, because you do not need to specify the datatype of your variables. Python will recognise it automatically. However, with low-level languages, like C, you need to specify the type of your data. Therefore, to enable efficient computing and connection to low-level languages, Numpy uses a special **dtype** object to define its arrays.

sample2_np.dtype

np.array([[1.1,2.2],[3.3,4.4]]).dtype

Full list of Numpy datatypes can be found here: [numpy.org/devdocs/user/basics.types.html](https://numpy.org/devdocs/user/basics.types.html). **Astype()** is an important function in Numpy. It can be used to change the **dtype** of an array. It also works with Pandas dataframes.

sample2_np.astype('float64') # Notice the dots.

Note that if you transform floats to integers, they will be truncated.

np.array([[3.4,2.3],[4.5,2.1]]).astype('int64')

*Vectorisation* is an important concept in Numpy. It means that you can do operations to whole arrays without using for loops. This is essential in many kinds of machine learning operations.

sample3_np = np.random.normal(size = (2,3))

sample3_np

sample3_np*sample3_np

Because of the vectorisation, for example simulations are very easy to do in Numpy.

coin_np = np.random.randint(0,2,200)

coin_np = np.where(coin_np>0,1,-1)

import matplotlib.pyplot as plt

plt.style.use('bmh')

plt.plot(coin_np.cumsum())

Dividing with an array works also element-wise.

1 / np.array([[1,2],[3,4]])

Broadcasting means that the operation of a smaller array is repeated through the larger array.

sample4_np = np.array([[1,2,3],[4,5,6]])

sample4_np + [1,1,1]

Comparison between arrays return a boolen ndarray.

rand1_np = np.random.normal(size=(2,2))
rand2_np = np.random.normal(size=(2,2))

rand1_np > rand2_np

Slicing works efficiently with Numpy.

rand3_np = np.random.randint(1,10,(3,4))

rand3_np

rand3_np[1]

Note that the starting value of a slice is not included. Overall, slicing with multi-dimensional arrays is something that needs practice. Experiment with different multidimensional arrays to learn the details of slicing.

rand3_np[1,2:4]

Broadcasting is also applied, when assigning values to slices

rand3_np[1] = 1

rand3_np



You can use booleans to pick values that satisfy a certain criteria. Notice that the result here is transformend as a one-dimensional array.

rand3_np[rand3_np < 5]

You can use also lists to select specific rows/columns.

rand3_np[[0,2]]

rand3_np[:,[0,2]]

You can easily reshape an array with **reshape**.

reshaped_np = np.arange(12).reshape((3,4))

reshaped_np

There is a special attribute **T** that can be used to transpose an array.

reshaped_np.T

Numpy has many other matrix operation functions also. The full list can be found here: [numpy.org/doc/stable/reference/routines.linalg.html](https://numpy.org/doc/stable/reference/routines.linalg.html)

np.dot(reshaped_np,reshaped_np.T) # Inner product

You can also calculate the product like this:

reshaped_np.dot(reshaped_np.T)

np.cross(reshaped_np[:2,:2],reshaped_np[:2,:2].T)

Numpy uses for matrix operations low-level libraries, so they are as efficient as the same operations, for example, in Matlab. 

square_np = np.random.normal(0,1,(3,3))

square_np

np.linalg.inv(square_np)

Here is an example of calculation errors that happens now and then with computers. The off-diagonal values should be exactly zero.

np.dot(square_np,np.linalg.inv(square_np))

Singular value decomposition.

np.linalg.svd(square_np)

There is also **swapaxes** that can be used to swap. So, for two-dimensinal arrays it is the same as transpose. But for higher-dimensional arrays, it gives more possibilities.

more_np = np.random.randint(1,10,size=(2,2,3))

more_np

more_np.swapaxes(0,2)

Numpy has many convenient functions for element-wise operations, called universal functions. Let's look some of them. The full list can be found here: [numpy.org/doc/stable/reference/ufuncs.html](https://numpy.org/doc/stable/reference/ufuncs.html)

np.sqrt(more_np)

np.exp(more_np)

array1_np = np.random.randint(1,10,size=(2,3))
array2_np = np.random.randint(1,10,size=(2,3))

array1_np

array2_np

np.maximum(array1_np,array2_np)

Numpy **where** is a very important function if you want to do conditional operations with arrays.

a_array_np = np.random.randint(1,10,(2,3))
b_array_np = np.random.randint(1,10,(2,3))
bool_np = np.array([[True,False,True],[False,True,False]])

a_array_np

b_array_np

np.where(bool_np,a_array_np,b_array_np)

There is also a sorting function in Numpy. At default, it sorts values along the last axis:

np.sort(b_array_np)

np.sort(b_array_np,axis=0)

np.sort(b_array_np,axis=None)

**Unique** is a Numpy version of **set**.

large_np = np.random.randint(1,10,size=(6,6))

large_np

np.unique(large_np)



#### Scikit-learn

Scikit-learn is a multi-purpose machine learning library. It has modules for many different machine learning approaches. It is not the best library in any machine learning field but very good at most of them. Also, all the approaches use the common workflow approach of the library. Thus, by learning to do one machine learning analysis, you learn to do them all.

Scikit-learn has libraries for classification, regression, clustering, dimensionality reduction, model selection and preprocessing. It also has an extensive library of methods for data pre-processing.

A very convenient feature in Scikit-learn is **pipeline** that you can use to construct full workflows of machine learning analyses.

There should be now difficulties to install Scikit-learn. With Python/Pip you just execute **pip install scikit-learn** and with Anaconda you just install it from the menu (or use **conda install scikit-learn** in the command line). (Actually, you should not need to do that as Scikit-learn is installed in Anaconda by default.)

Again, the best way to learn Scikit-learn is by going through examples. Thus, more  details are in the following examples.

Our sample dataset consists of few key financials of the largest US companies. Let's load it.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('bmh')

Example data from [www.kaggle.com/c/companies-bankruptcy-forecast](https://www.kaggle.com/c/companies-bankruptcy-forecast)

table_df = pd.read_csv('ml_data.csv')[['Attr1','Attr8','Attr21','Attr4',
                                       'Attr5','Attr29','Attr20','Attr12',
                                       'Attr15','Attr6','Attr24','Attr44','Attr47','class']]

table_df

table_df.rename({'Attr1' : 'ROA','Attr8' : 'Leverage','Attr21' : 'Sales-growth',
                 'Attr4' : 'Current ratio','Attr5' : 'Quick ratio','Attr29' : 'Log(Total assets)',
                 'Attr20' : 'Inventory*365/sales','Attr12' : 'Gross_prof/st_liab',
                'Attr15' : 'Total_liab*365/(gross_prof+depr)','Attr6' : 'Ret_earnings/TA',
                'Attr24' : 'Gross_prof/TA(3yr)','Attr44' : 'Receiv*365/sales',
                'Attr47' : 'Inv*365/CoGS'},axis=1,inplace=True)

table_df = table_df.clip(lower=table_df.quantile(0.01),upper=table_df.quantile(0.99),axis=1)

table_df.hist(figsize=(14,14))
plt.show()

X = table_df.drop(['ROA','class'],axis=1)

y = table_df['ROA']

from sklearn.model_selection import train_test_split

Let's make things difficult for OLS (very small train set).

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.995, random_state=1)

len(X_train)

#### Linear model

Although Scikit-learn is a ML library, it is possible to do a basic linear regression analysis with it. (All ML methods are statistical methods. The separation between them is artificial.)

import sklearn.linear_model as sk_lm

model = sk_lm.LinearRegression()

model.fit(X_train,y_train)

model.coef_

model.intercept_

model.score(X_test,y_test)

fig, axs = plt.subplots(6,2,figsize=(15,20))
for ax,feature,coef in zip(axs.flat,X_test.columns,model.coef_):
    ax.scatter(X_test[feature],y_test,alpha=0.5)
    ax.plot(X_test[feature],model.predict(X_test),'r.',alpha=0.5)
    ax.set_title(feature)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,model.predict(X_test))

#### Ridge regression

Ridge regression counters overfitting by adding a penalty on the size if the coefficients of the standard linear regression model.

We can optimise the alpha parameter of the error function automatically using **RidgeCV**.

alpha_set = np.logspace(-10,10,21)

ridgecv = sk_lm.RidgeCV(alphas = alpha_set,cv=5, scoring = 'neg_mean_squared_error', normalize = True)

ridgecv.fit(X_train,y_train)

As you can see, the coefficients have decreases. But only a little.

ridgecv.coef_

ridgecv.intercept_

ridgecv.alpha_

Ridge regression decreases  the variation of predictions.

fig, axs = plt.subplots(3,2,figsize=(15,15))
for ax,feature,coef in zip(axs.flat,X_test.columns,model.coef_):
    ax.scatter(X_test[feature],y_test,alpha=0.3)
    ax.plot(X_test[feature],ridgecv.predict(X_test),'r.',alpha=0.3)
    ax.set_title(feature)

mean_squared_error(y_test,ridgecv.predict(X_test))

#### The Lasso

Let's try next the lasso.



alpha_set = np.logspace(-5,5,21)

lassocv = sk_lm.LassoCV(alphas = None,cv=10,max_iter=100000, normalize = True)

lassocv.fit(X_train,y_train)

As you can see, the coefficients have decreases. But only a little.

lassocv.coef_

lassocv.intercept_

lassocv.alpha_

Ridge regression decreases  the variation of predictions.

fig, axs = plt.subplots(3,2,figsize=(15,15))
for ax,feature,coef in zip(axs.flat,X_test.columns,model.coef_):
    ax.scatter(X_test[feature],y_test,alpha=0.3)
    ax.plot(X_test[feature],lassocv.predict(X_test),'r.',alpha=0.3)
    ax.set_title(feature)

mean_squared_error(y_test,lassocv.predict(X_test))

As you can see from the results, the Lasso and ridge regression are usefuly only when n is close to p.



lasso_model = sk_lm.Lasso(alpha = 0.01,max_iter=100000, normalize = True)

lasso_model.fit(X_train,y_train)

As you can see, the coefficients have decreases. But only a little.

lasso_model.coef_

lasso_model.intercept_

Ridge regression decreases  the variation of predictions.

fig, axs = plt.subplots(6,2,figsize=(15,20))
for ax,feature,coef in zip(axs.flat,X_test.columns,model.coef_):
    ax.scatter(X_test[feature],y_test,alpha=0.3)
    ax.plot(X_test[feature],lasso_model.predict(X_test),'r.',alpha=0.3)
    ax.set_title(feature)

mean_squared_error(y_test,lasso_model.predict(X_test))

#### Bayesian ridge regression

