## Deep learning example from Accounting/Finance

The following example demonstrates a simple example of deep learning that uses accounting/finance data. It also demonstrates, how to implement a deep learning model to traditional structured data. However, it also shows how deep learning is usually not the best option for structured data with relatively small datasets (<100k observations). Deep learning models perform better with large unstructured datasets.

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

The data has a little under 20k observations. The variables are different financial ratios and board characteristics of S&P1500 companies.

compu_df = pd.read_csv('_data.txt',delimiter='\t')

compu_df

We use only variables with few missing values, because the benefits of deep learning are seen in large datasets. The variables are different financial ratios (Compustat). There are also many heavily correlating variables included, but this should not be a serious issue with neural networks.

variables = ['at', 'bkvlps','capx', 'ceq', 'csho', 'cstk', 'dlc', 'dltt', 'dvc', 'ebit',
       'ibc', 'icapt', 'lt', 'ni', 'pstk', 'pstkl','pstkrv', 're', 'sale', 'seq', 'costat', 'prcc_c',
       'prcc_f', 'sic', 'mkvalt','tobin', 'yld', 'age', 'tridx', 'mb',
       'cap_int', 'lvg', 'roa', 'roe', 'roi']

compu_df[variables].isna().sum()

compu_df[variables] = compu_df[variables].clip(lower=compu_df[variables].quantile(0.01),
                                               upper=compu_df[variables].quantile(0.99),axis=1)

compu_df['current_roa'] = compu_df['roa']

Lag everything else

compu_df[variables] = compu_df.groupby(['conm']).shift()[variables]

We have to drop all missing values, because othwerise we can not optimize the network using gradient descent algorithm.

I add industry, SP500 dummy and year separately as I do not want to winsorize or lag these variables.

compu_df[variables + ['fyear','ind','sp500','current_roa']] = compu_df[variables + ['fyear','ind','sp500','current_roa']].dropna()

compu_df[variables + ['fyear','current_roa']].head(30)

I remove the first year (2008), because we do not have any observations there due to the lag procedure.

compu_df = compu_df[compu_df['fyear'] > 2008.]

We try to predict current ROA with the last year's variable values.

y_df = compu_df['current_roa']

x_df = compu_df[variables + ['fyear','ind','sp500']]

Train/test split

from sklearn.model_selection import train_test_split

Tensoflow does not like Pandas dataframes, so I change them to Numpy array.s

# Split data into training and test sets
x_train, x_test , y_train, y_test = train_test_split(x_df.values, y_df.values, test_size=0.20, random_state=1)

type(x_train)

len(x_train), len(x_test)

Let's check that there is no missing values any more.

compu_df[variables+['current_roa','fyear']].isna().sum()

### Densely connected network
Let's build a traditional densely connected neural network. We could also use recurrent or LSTM networks, but we would need to reorganize data in that case.

![image.png](./images/feed_forward.png)!

One way to define a neural network with Keras is a single **Sequential**-command that has the layers in a list as a parameter. The densely connected layers have **ReLU** as an activation function. The last layer has one neuron, because we want to have a single valua as an output (current ROA). There is also no activation function, because we want to have linear output. There is also a dropout layer to counter overfitting. For the first layer, we need to define the shape of our input.

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu',input_shape = (38,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)])

You can check your model with the **summary()** -function. The model has 203 530 parameters.

model.summary()

With **compile()**, we build our neural network to be ready for use. For regerssion problems, **MSE** is the correct loss function. We measure our performance with **Mean Absolute Error**, because it is easier to interpret than MSE.

model.compile(loss='mse',metrics=['mae'])

Next, we feed the training data to our model and train it using back-propagation. Everything is automatic, so, we do not need to worry about the details. The training data accuracy rises to 0.986 = 98.6%. However, true performance needs to be evaluated using test data. We can save to **history** information about the training process. The model is trained with batches of 64 images. So, to go through all the images, we need 938 rounds (the last batch has 32 images). One epoch is one round of going through all the data.

history = model.fit(x_train,y_train,epochs=150,batch_size=1024,validation_split=0.1,verbose=False)

The following code plots the progress of training. Within the code is info for different commands.

plt.style.use('bmh') # We select as a plot-style 'bmh' that is in my opinion usually the prettiest.
burnout = 25
epochs = range(1, len(history.history['val_mae']) + 1) # Correct x-axis values (epochs)
plt.plot(epochs[burnout:], history.history['val_mae'][burnout:], 'r--',label='Validation accuracy') # Plot epochs vs. accuracy
plt.plot(epochs[burnout:], history.history['mae'][burnout:], 'b--',label='Train accuracy') # Plot epochs vs. accuracy
plt.legend()
plt.title('Accuracy') # Add title
plt.figure() # Show the first figure. Without this command, accuracy and loss would be drawn to the same plot.
plt.plot(epochs[burnout:], history.history['val_loss'][burnout:], 'r--',label='Validation loss') # Plot epochs vs. loss
plt.plot(epochs[burnout:], history.history['loss'][burnout:], 'b--',label='Train loss')
plt.title('Loss') # Add title
plt.show() # Show everyhting

**Evaluate()** can be used to evaluate the model with the test data. Acccuracy with the test data is 0.052.

test_loss,test_acc = model.evaluate(x_test,y_test)

test_acc

Let's compare the performance to a linear model.

import sklearn.linear_model as sk_lm

We define our LinearRegression object.

model = sk_lm.LinearRegression()

**fit()** can be used to fit the data.

model.fit(x_train,y_train)

**coef_** -attribute has the coefficients of each variable and **intercept_** has the intercept of the linear regression model.

model.coef_

model.intercept_

**score()** can be used to measure the coefficient of determination of the trained model. How much our variables are explaining of the variation of the predicted variable.*

model.score(x_test,y_test)

Mean absolute error.

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,model.predict(x_test))

As expected, the linear model performs better for this data.

