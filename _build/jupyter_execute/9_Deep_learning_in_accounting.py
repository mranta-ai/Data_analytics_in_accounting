## Deep learning in accounting

### Neural networks
Neural networks are a large class of models and learning methods. The most traditional is the single hidden layer back-propagation network or a single layer perceptron.

![trad_nnetwork](./images/trad_nnetwork.svg)

The above networks belongs to a class of networks called feed-forward neural networks. The name comes from the fact that we calculate linear combinations of inputs at different layers, pass the results to a nonlinear activation function and feed the value forward. By combining these nonlinear functions of linear combinations into a network, we get a very powerful non-linear estimator.

Basically, deep learning just means that we add many layers to the network. (The paradigm of deep learning is a little bit *deeper* than that, but we do not go into details at this point.)

Despite the hype surrounding neural networks, they are no different from other nonlinear statistical models. They are just networks of neurons that calculate linear combinations of input values and input these linear combinations to nonlinear activation functions. The result is a powerful learning method, with widespread applications in many fields. The most successful neural network architectures in computer vision are convolutional neural networks that are introduced in more detail below. 

### Traditional feed-forward neural networks

The name of this architecture comes from the way information flows through the network from input **x** to the output **y**. At every neuron, information flows towards output without any feedback connections.

If we add feedback connections to feed-forward networks, they are called recurrent neural networks. 

Usually, feed-forward networks are considered to consist of a sequency of dense layers that take vectors as inputs. However, sometimes the type is considered more general and, for example, convolutional neural networks are considered to be a special type of feed-forwarnd neural network.

Just by adding together neurons that calculate linear combinations of vectors does not give us more explanatory power when compared to traditional methods like linear regression. The key element is the activation function of neurons, which enables the networks to estimate nonlinear structures.

As can be observed from the figure below, neural network needs a lot of data as the number of parameters grows exponentially with the complexity of the network. Why? Because we need to have more data than parameters or the model will overfit severely.

![two_ann](./images/Two_layer_ann.svg)

### Convolutional neural networks
In the computer vision context, the input for convolutional neural networks (CNN) is a multi-channeled image (instead of a vector that is commonly used in standard feed-forward neural networks), i.e. a 3D-tensor (several channels of 2D images).

The network consists of convolution layers and pooling layers.

![image.png](./images/conv_structure_1.png)

The convolution layers filter feature maps (channels in the original image) with small filters that are slid through the maps.
* Convolution  = filtering --> A dot product between the filter and a portion of the image (plus bias).
* The filter is slid through an image (every channel of the image).
* As a result, we get a slightly smaller "image" of dot products.
* The convolution layer is the main building block of CNNs

The fundamental difference between a densely connected layer and a convolution layer is that dense layers learn global structures in their input feature maps (channels), whereas convolution layers learn local patterns. This is useful in computer vision problems because, in the case of images, these local patterns can be located anywhere in the image. Moreover, CNNs have usually chains of convolutional layers, which causes these learned patterns to become more "complex" the deeper we are in the CNN. The first convolutional layers learn arcs, lines etc. and later layers connect these to circles and other more complex structures (, of course depending on the computer vision task at hand). A first convolutional layer learns small and simple patterns, a second convolutional layer learns patterns that are constructed from the patterns of the first layer, and so on.

These characteristic give CNNs interesting properties. For example, the learned patterns are translation invariant. A certain pattern learned at a certain location can be recognised anywhere in an image, a key property for computer vision tasks. A traditional feed-forward network would have to learn a certain pattern anew for every location in an image. This makes CNNs data-efficient; they need fewer training samples to learn representations that have generalisation power.

#### ReLUs
CNNs usually use rectified linear units (ReLU) as activation functions to add nonlinearity, just like traditional densely connected neural networks. Without a non-linear activation function, the network would be linear (no matter how many layers, a linear combination of linear combinations is still a linear combination).

![image.png](./images/relu.svg)

#### Pooling layers
* Its function is to progressively reduce the spatial size of the representation to reduce the number of parameters and computation in the network. Pooling layer operates on each feature map independently.
* The most common approach used in pooling is max pooling.

![image.png](./images/Max_pooling.png)

#### Feature maps
As I mentioned, convolutional layers operate over 3D tensors, called feature maps, with two spatial axes (2D image) as well as a depth axis (also called the channels axis). For a usual RGB image, the dimension of the depth axis is  3 (red, green and blue). For a black-and-white picture, there is no depth axis, but one feature map representing different levels of grey (or the third dimension has unit length). The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a 3D tensor: it has a width and a height that depend on the convolutional filter used. The depth is a parameter of the model, which increases when moving from left to right in a CNN. However, this does not increase the number of parameters in a model, because CNNs usually use pooling layers to decrease the size of feature maps. The channels no longer stand for specific colours as in an RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept "presence of a face in the input" for instance.

![image.png](./images/feature_maps.png)

#### Best pretrained models
It is often beneficial to use pre-trained networks in practical computer vision applications. Pretrained networks have their parameters trained with very large datasets using HPC capabilities. A very common dataset is Imagenet that has over 14 million images ([www.image-net.org](https://www.image-net.org)). Below are some popular pre-trained networks that have proven to be very efficient, according to the ImageNet Large Scale Visual Recognition Challenge -winners.

**ImageNet Large Scale Visual Recognition Challenge -winners**

Previous winners, with shallow neural networks, achieved around 25 % error rate. The examples below are all deep convolutional neural networks.

* 2012 - Alexnet - 8 layers - 16 % error rate - a much deeper structure and utilised GPUs during training
* 2013 - ZF-net - 8 layers - 12 % error rate
* 2014 - VGG - 19 layers - 7.3 % error rate
* 2014 - GoogleNet - 22 layers - 6.7 % error rate

Human error rate around 5 %

* 2015 - ResNet - 152 layers - 3.6 % error rate - innovation: a residual learning framework that improves the training of very deep networks
* 2016 - Ensemble of previous models - 3.0 % error rate
* 2017 - SENet - 2.25 % error rate

### Recurrent neural networks

Recurrent neural networks excel at analysing sequences, lime time series and language. The basic architecture is like feed-forward neural network, but added with feedback connections back to the sending neuron.

![rnn](./images/rnn.svg)

The original recurrent neural network architecture achieved acceptable results with sequences. However, its performance decreases quickly when the sequence becomes longer. To overcome this problem, long short-term memory networks were invented. They try keep the training of a netwrok efficient with longer sequences. Until the invention of transformes architecture, LSTM networks were the state-of-the-art for tasks like natural language processing.
![lstm](./images/lstm.svg)

As you can see from the figure, the exact theory of LSTM networks is difficult and we skip it at this point.


### Tensorflow

Tensorflow is a very popular and extensive deep learning library. It is also somewhat difficult to learn. For our purposes, it is enouqh that we learn the basics of the Keras module included in Tensorflow.

With Keras, it is extremely easy to build machine learning models. It is like building with Lego bricks. You just add the layers of your network, define the details of each layer, select the error function and the output function, and that's it.

If you want to replicate the Tensorflow code of this book, I strongly suggest that you **install Tensorflow within Anaconda environment**. At least, if you want to use gpu with Tensorflow. To enable the Nvidia-GPU support, you need to install also the CUDA-libraries provided by Nvidia. However, it is insanely difficult to get everything working in the GPU drivers-CUDA-Tensorflow -axis. Anaconda does everything automatically for you.

import tensorflow as tf

With the following commands, you can check what kind of computing units are available. Of course, you are looking for GPU-units to speed up computations.

tf.config.list_physical_devices()

So, I have Quadro P5200 available for calculations. It will speed up calculations a lot when compared to CPU. If you do not have an Nvidia GPU available for calculations, some of the steps below will be very slow to calculate.

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

One way to define a neural network with Keras is a single **Sequential**-command that has the layers in a list as a parameter. The densely connected layer has **ReLU** as an activation function. Because we want to categorise ten digits, **softmax** is the correct activation function for the last layer. Notice how the last layer has ten outputs, one for each digit. The **Flatten()**-layer transforms the 28 x 28 -image to a vector of size 784.

With Keras, we can also build the network using sequential **add()**-commands. We first define the **Sequential()**-type and then add the layers with the **add()** -function.

It is much easier to understand how Keras works by following an example. So, lets work through a detailed example using Keras.



### The MNIST dataset

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used as the first test for computer vision models. These are 28x28 pixel images that are anti-aliased from the original NIST black and white images, and thus, grayscale images with 256 levels. The database contains 60,000 training images and 10,000 testing images.

![MNIST](./images/mnist.png)

**Keras.dataset** has the MNIST dataset, so we avoid the hassle of downloading and preprocessing the dataset by ourselves.

mnist = tf.keras.datasets.mnist

With **load_data()**, we can load the train and test datasets as **Numpy arrays**.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

type(x_train)

We have 60000 images in the training set and 10000 images in the test set.

len(x_train), len(x_test)

Let's check what kind of data we have. Matplotlib has a function to plot images. First, we load the library. With **plt.imshow()**, we can plot the image. We need to set **cmap=gray** to get the correct grayscale image.

import matplotlib.pyplot as plt

plt.imshow(x_train[0],cmap='gray')

y_train[0]

The originals are greyscale images with each pixel having a value from 0 to 255. We normalize them to values between 0 and 1 (neural networks like small values).

x_train, x_test = x_train / 255.0, x_test / 255.0

### Densely connected network
First, we fit a traditional densely connected feed-forward neural network to the data.

![image.png](./images/feed_forward.png)!

Our data is a 3D-tensor of the form 60000 images x 28 pixels x 28 pixels. Sometimes we need to make the transform (60000,28,28) --> (60000,28,28,1) and sometimes not. If you get an error in the model.fit -step, run the code below.

x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

Currently, our labels are integers from 0 to 9. We need to transorm them to binary classes. For example 1-->(0,1,0,0,0,0,0,0,0,0) and 9-->(0,0,0,0,0,0,0,0,0,1). This can be done with the **to_categorical()** -function in Keras.utils.

train_labels = tf.keras.utils.to_categorical(y_train)

test_labels = tf.keras.utils.to_categorical(y_test)

One way to define a neural network with Keras is a single **Sequential**-command that has the layers in a list as a parameter. The densely connected layer has **ReLU** as an activation function. Because we want to categorise ten digits, **softmax** is the correct activation function for the last layer. Notice how the last layer has ten outputs, one for each digit. The **Flatten()**-layer transforms the 28 x 28 -image to a vector of size 784.

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(10,activation='softmax')
])

You can check your model with the **summary()** -function. The model has 203 530 parameters.

model.summary()

With **compile()**, we build our neural network to be ready for use. For classification problems, **categorical_crossentropy** is the correct loss function. We measure our performance with accuracy. It is just the percentage of correct classifications.

model.compile(loss='categorical_crossentropy', metrics = ['accuracy'])

Next, we feed the training data to our model and train it using back-propagation. Everything is automatic, so, we do not need to worry about the details. The training data accuracy rises to 0.986 = 98.6%. However, true performance needs to be evaluated using test data. We can save to **history** information about the training process. The model is trained with batches of 64 images. So, to go through all the images, we need 938 rounds (the last batch has 32 images). One epoch is one round of going through all the data.

history = model.fit(x_train,train_labels,epochs=5,batch_size=64)

The following code plots the progress of training. Within the code is info for different commands.

plt.style.use('bmh') # We select as a plot-style 'bmh' that is in my opinion usually the prettiest.
acc = history.history['accuracy'] # The evolution of accuracy to a list.
loss = history.history['loss'] # The evolution of loss to a list.
epochs = range(1, len(acc) + 1) # Correct x-axis values (epochs)
plt.plot(epochs, acc, 'r--') # Plot epochs vs. accuracy
plt.title('Accuracy') # Add title
plt.figure() # Show the first figure. Without this command, accuracy and loss would be drawn to the same plot.
plt.plot(epochs, loss, 'b--') # Plot epochs vs. loss
plt.title('Loss') # Add title
plt.show() # Show everyhting

**Evaluate()** can be used to evaluate the model with the test data. Acccuracy with the test data is 0.974

test_loss,test_acc = model.evaluate(x_test,test_labels)

test_acc

### Convolutional neural network

Identifying the digits correctly is a computer vision problem. So, we should expect that convolutional neural networks would perform better. Thus, we build a simple CNN to identify the digits.

![image.png](./images/cnn_example.gif)

With Keras, we can also build the network using sequential **add()**-commands. We first define the **Sequential()**-type and then add the layers with the **add()* -function.

model_cnn = tf.keras.models.Sequential()

A convolutional layer with 32 feature maps and a 3x3 -filter: The activation is again **ReLU**. For the first layer, we need to define the format of the input data, in this case, 28 x 28 -pixel images.

model_cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))

A max-pooling layer to decrease the size of the feature maps. The maximum values are selected from a 2 x 2 window.

model_cnn.add(tf.keras.layers.MaxPooling2D((2,2)))

Again, a convolutional layer. Notice how the number of feature maps increases. This is typical for CNN architectures.

model_cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))

A second max-pooling layer.

model_cnn.add(tf.keras.layers.MaxPooling2D((2,2)))

A third convolutional layer.

model_cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'))

**Flatten()** -layer transforms the 2D feature maps to a 1D vector so that we can feed it to an ordinary densely-connected layer.

model_cnn.add(tf.keras.layers.Flatten())

One densely-connected layer before the output-layer.

model_cnn.add(tf.keras.layers.Dense(64,activation = 'relu'))

The output-layer has ten neurons that give probabilities for each digit. **Softmax** is the correct activation function for categorical predictions.

model_cnn.add(tf.keras.layers.Dense(10,activation = 'softmax'))

The **summary()** -function shows that our model has 93 322 parameters. Thus, much less than the previous densely-connected traditional neural network.

model_cnn.summary()

Again, we compile our model...

model_cnn.compile(loss='categorical_crossentropy', metrics = ['accuracy'])

...and train it.

Although our model has much fewer parameters, the performance with the training data is much better. Now, the accuracy is 0.994. Let's see how it performs with the test data...

history = model_cnn.fit(x_train,train_labels,epochs=5,batch_size=64)

Again we plot the progress from **history**.

plt.style.use('bmh') # We select as a plot-style 'bmh' that is in my opinion usually the best.
acc = history.history['accuracy'] # The evolution of accuracy to a list.
loss = history.history['loss'] # The evolution of loss to a list.
epochs = range(1, len(acc) + 1) # Correct x-axis values (epochs)
plt.plot(epochs, acc, 'r--') # Plot epochs vs. accuracy
plt.title('Accuracy') # Add title
plt.figure() # Show the first figure. Without this command, accuracy and loss would be drawn to the same plot.
plt.plot(epochs, loss, 'b--') # Plot epochs vs. loss
plt.title('Loss') # Add title
plt.show() # Show everyhting

With **evaluate()**, we can check the performance with the test data. We achieve much better accuracy of 0.989 with much fewer parameters.

test_loss,test_acc = model_cnn.evaluate(x_test,test_labels)

test_acc

### Example - dogs and cats

For this example, we need to load the data ourselves that is somewhat laborious. We use image classification data from [www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats). Kaggle organises ML-competitions, and in this competition, the task is to distinguish dogs from cats in images.

![Pinto](./images/pinto.png)

First, we load some libraries that are needed to manipulate the image files.

import os,shutil

I have the original training data in the "original_data" folder (under the work folder). You can download the original data from [www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats).

files = os.listdir('./original_data')

The total number of dog and cat images is 25 000.

len(files)

We do this training "by-the-book" by dividing the data to train, validation and test parts. The validation data is used to finetune the hyperparameters of a model. With separate validation data, we avoid using the hyperparameter optimisation wrongly to optimise the test data performance. Below is an example of a dataset-split that uses 3-fold cross-validation.
![Validation](./images/Train-Test-Validation.png)

The following commands build different folders for the training, validation and test data.

os.mkdir('train')
os.mkdir('validation')
os.mkdir('test')

Under the training, validation and test -folders we make separate folders for the dog and cat pictures. This makes it much easier to use Keras data-generation function as it can automatically collect observations of different classes from different folders. **os.path.join()** -function makes it easy to build directory structures. You add the "parts" of the directory structure, and it will add automatically slashes when needed.

# Get the current work directory
base_dir = os.getcwd()

# Dogs
os.mkdir(os.path.join(base_dir,'train','dogs'))
os.mkdir(os.path.join(base_dir,'validation','dogs'))
os.mkdir(os.path.join(base_dir,'test','dogs'))
# Cats
os.mkdir(os.path.join(base_dir,'train','cats'))
os.mkdir(os.path.join(base_dir,'validation','cats'))
os.mkdir(os.path.join(base_dir,'test','cats'))

Next, we copy the files to correct folders. We use only part of the data to speed up calculations: 3000 images for the training, 1000 images for the validation and 1000 images for the testing. The first command in each cell constructs a list of correct filenames. It uses Python's list comprehension, that is a great feature in Python.

Let's analyse the first one (**fnames = ['dog.{}.jpg'.format(i) for i in range(1500)]**):

When we put a for loop inside square brackets, Python will generate a list that has the "rounds" of a loop as values in the list.

**'dog.{}.jpg'.format(i)** - This is the part that will be repeated in the list so that the curly brackets are replaced by the value of **i**.

**for i in range(1500)** - This will tell what values are inserted in **i**. **range(1500)** just means values from 0 to 1500.

More information about list comprehension can be found from https://docs.python.org/3/tutorial/datastructures.html (section 5.1.3)

# Train dogs
fnames = ['dog.{}.jpg'.format(i) for i in range(1500)]
for file in fnames:
    src = os.path.join(base_dir,'original_data',file)
    dst = os.path.join(base_dir,'train','dogs',file)
    shutil.copyfile(src,dst)

# Validation dogs
fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for file in fnames:
    src = os.path.join(base_dir,'original_data',file)
    dst = os.path.join(base_dir,'validation','dogs',file)
    shutil.copyfile(src,dst)

# Test dogs
fnames = ['dog.{}.jpg'.format(i) for i in range(2000,2500)]
for file in fnames:
    src = os.path.join(base_dir,'original_data',file)
    dst = os.path.join(base_dir,'test','dogs',file)
    shutil.copyfile(src,dst)

# Train cats
fnames = ['cat.{}.jpg'.format(i) for i in range(1500)]
for file in fnames:
    src = os.path.join(base_dir,'original_data',file)
    dst = os.path.join(base_dir,'train','cats',file)
    shutil.copyfile(src,dst)

# Validation cats
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for file in fnames:
    src = os.path.join(base_dir,'original_data',file)
    dst = os.path.join(base_dir,'validation','cats',file)
    shutil.copyfile(src,dst)

# Test cats
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for file in fnames:
    src = os.path.join(base_dir,'original_data',file)
    dst = os.path.join(base_dir,'test','cats',file)
    shutil.copyfile(src,dst)

Next, we check that everything went as planned. The dog folders should have 1500,500 and 500 images and similarly to the cat folders.

# Check the dog directories
print(len(os.listdir(os.path.join(base_dir,'train','dogs'))))
print(len(os.listdir(os.path.join(base_dir,'validation','dogs'))))
print(len(os.listdir(os.path.join(base_dir,'test','dogs'))))

# Check the cat directories
print(len(os.listdir(os.path.join(base_dir,'train','cats'))))
print(len(os.listdir(os.path.join(base_dir,'validation','cats'))))
print(len(os.listdir(os.path.join(base_dir,'test','cats'))))

#### Simple CNN model

As our preliminary model, we test a basic CNN model with four convolutional layers and four max-pooling layers followed by two dense layers with 12544 (flatten) and 512 neurons. The output layer has one neuron with a sigmoid activation function. So, the output is a prediction for one of the two classes.

First, we need the **layers** and **models** -modules from Keras.

from tensorflow.keras import layers
from tensorflow.keras import models

Next, we define a sequential model and add layers using the **add()**-function.

model = models.Sequential()

The input images to the network are 150x150 pixel RGB images. The size of the convolution-filter is 3x3, and the layer produces 32 feature maps. The ReLU activation function is the common choice with CNNs (and many other neural network types).

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))

A max-pooling layer with a 2x2 window.

model.add(layers.MaxPooling2D((2, 2)))

Notice how the number of feature maps is increasing.

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

Overall, we have almost 7 million parameters in our model, which is way too much for a training set with 3000 images. The model will overfit as we will soon see from the results.
![First_CNN](./images/nn.png)

model.summary()

from tensorflow.keras import optimizers

Next, we compile the model. Because we have now two classes, "binary_crossentropy" is the correct loss_function. There are many gradient descent optimisers available, but usually, RMSprop works very well. More information about RMSprop can be found here: https://keras.io/api/optimizers/rmsprop/.

![grad_desc](./images/Gradient_descent.gif)

We measure performance with the accuracy metric.

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(),metrics=['acc'])

To get images from a folder to a CNN model can be a very tedious task. Luckily, Keras has functions that make our job much more straightforward.

**ImageDataGenerator** is a Python generator that can be used to transform images from a folder to tensors that can be fed to a neural network model.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

We scale the pixel values from 0-255 to 0-1. Remember: neural networks like small values.

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

We change the size of the images to 150 x 150 and collect them in 25 batches. Basically, we feed (25,150,150,3)-tensors to the model. As you can see, the function automatically recognises two different classes. It is because we placed the cat and dog images to two different folders. We have to make separate generators for the training data and the validation data.

train_generator = train_datagen.flow_from_directory(os.path.join(base_dir,'train'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(os.path.join(base_dir,'validation'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

We use a little bit longer training with 30 epochs. Instead of input data, we now give the generators to the model. Also, we separately define the validation generator and validation testing steps. With 25 image batches and 120 steps per epoch, we go through all the 3000 images. To **history**, we save the training progress details.

history = model.fit(train_generator,
                              steps_per_epoch=120,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=40)

Let's check how did it go. In a typical overfitting situation, training accuracy quickly rises to almost 1.0 and validation accuracy stalls to a much lower level. This is also the case here. The training accuracy is 0.984, and the validation accuracy is around 0.72.
But still, not that bad! The model recognises cats and dogs correctly from the images 72 % of the time.

![Overfitting](./images/Overfitting.svg)

import matplotlib.pyplot as plt # Load plotting libraries
plt.style.use('bmh') # bmh-style is usually nice
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training accuracy')
plt.plot(epochs, val_acc, 'b--', label='Validation accuracy')
plt.legend() # Matplotlib will automatically position the legend in a best possible way.
plt.figure() # This is needed to make two separate figures for loss and accuracy.
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.legend()
plt.show()

#### Augmentation and regularisation

Let's try to improve our model. Augmentation is a common approach to "increase" the amount of data. The idea of augmentation is to transform images slightly every time they are fed to the model. Thus, we are trying to create new information to the model to train on. However, we are not truly creating new information. Nevertheless, augmentation has proven to be an efficient way to improve results.

Image transformation can be implemented to the **ImageDataGenerator()**-function. There are many parameters that can be used to transform images. More information: [keras.io/api/preprocessing/image/](https://keras.io/api/preprocessing/image/)

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

Let's check what kind of images we are analysing.

# Image -module to view images
from tensorflow.keras.preprocessing import image

# We pick the 16th image from the train/dogs -folder.
img_path = os.path.join(base_dir,'train','dogs',os.listdir(os.path.join(base_dir,'train','dogs'))[16])

sample_image = image.load_img(img_path, target_size=(150, 150))

Below is an example image from the original dataset. The sixteenth image in our list.

sample_image

To use the Imagedatagenerator's **flow()**-function, we need to transform our image to a numpy-array.

sample_image_np = image.img_to_array(sample_image)
sample_image_np = sample_image_np.reshape((1,) + sample_image_np.shape)

The following code transforms images using *ImageDataGenerator()* and plots eight examples. As you can see, they are slightly altered images that are very close to the original image.

fig, axs = plt.subplots(2,4,figsize=(14,6),squeeze=True)
i=0
for ax,transform in zip(axs.flat,datagen.flow(sample_image_np, batch_size=1)):
    ax.imshow(image.array_to_img(transform[0]))
    i+=1
    if i%8==0:
        break

Next, we define the model. Alongside augmentation, we add regularisation to the model with a dropout-layer. The Dropout layer randomly sets input units to 0 with a frequency of **rate** (0.5 below) at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

We build our sequential model using **add()**-functions. The only difference, when compared to the previous model, is the dropout-layer after the flatten-layer (and the augmentation).

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

The dropout layer does not change the number of parameters. It is exactly the same as in the previous model.

model.summary()

The compile-step is not changed.

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(),metrics=['acc'])

We create the augmentation-enabled generators. Remember that the validation dataset should not be augmented!

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

The same dataset of 3000 training images and 1000 validation images.

train_generator = train_datagen.flow_from_directory(os.path.join(base_dir,'train'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(os.path.join(base_dir,'validation'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

Otherwise, the parameters to the **model.fit()** are the same as in the previous model, but we train the model a little bit longer. This is because regularisation slows down training.

history = model.fit(train_generator,
                              steps_per_epoch=120,
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=40)

As you can see from the following figure, overfitting has almost disappeared. The training and validation accuracy stay approximately at the same level through the training. The performance is also somewhat better. Now we achieve a validation accuracy of 0.77.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b--', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.legend()
plt.show()

#### Pre-trained model

Next thing that we could try is to use a pre-trained model that has its parameters already optimised using some other dataset. Usually, CNNs related to computer vision are pre-trained using Imagenet data (http://www.image-net.org/). It is a vast collection of labelled images.

We add our own layers after the pre-trained architecture. As our pre-trained model, we use VGG16

![VGG16](./images/vgg.png)

VGG16 is included in the **keras.applications** -module

from tensorflow.keras.applications import VGG16

When we load the VGG16 model, we need to set **weights=imagenet** to get pre-trained parameter weights. **include_top=False** removes the output layer with 1000 neurons. We want our output layer to have only one neuron (prediction for dog/cat).

pretrained_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))

VGG16 has 14.7 million parameters without the last layer. It also has two or three convolutional layers in a row. Our previous models were switching between a convolutional layer and a max-pooling layer.

pretrained_base.summary()

model = models.Sequential()

When we construct the model, we add the pre-trained VGG16-base first. Then follows a 256-neuron Dense-layer and a one-neuron output layer.

model.add(pretrained_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

Overall, our model has almost 17 million parameters. However, we will lock the pre-trained VGG16 base, which will decrease the number of trainable parameters significantly.

model.summary()

We want to use the pretrained Imagenet-weights, so, we lock the weights of the VGG16 -part.

pretrained_base.trainable = False

Now there is "only" two million trainable parameters.

model.summary()

Again, we use the augmentation of the training dataset.

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(os.path.join(base_dir,'train'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(os.path.join(base_dir,'validation'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

Compile- and fit-steps do not have anything new.

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(),metrics=['acc'])

history = model.fit(train_generator,
                              steps_per_epoch=120,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=40)

As you can see from the plots below, there is a small overfitting issue. The difference between the training accuracy and the validation accuracy increases slowly. However, the performance is excellent! Now our model can separate dogs from cats correctly 85 % of the time.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b--', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.legend()
plt.show()

#### Fine tuning

There is still (at least) one thing that we can do to improve our model. We can finetune our pre-trained VGG16 model by opening part of its' weights. As our VGG16 is now optimised for Imagenet data, the weights have information about features that are useful for many different types of images. By opening the last few layers of the model, we allow it to finetune those weights to features that are useful in separating dogs from cats in images.

First, we need to make our VGG16 model trainable again.

pretrained_base.trainable = True

Here is the summary of the VGG16 model again.

pretrained_base.summary()

Let' lock everything else, but leave the layers of **block5** to be finetuned by our dogs/cats images. The following code will go through the VGG16 structure, lock everything until 'block4_pool' and leave layers after that trainable.

set_trainable = False
for layer in pretrained_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

There are over 9 million trainable parameters, which can probably cause overfitting, but let's see.

model.summary()

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(),metrics=['acc'])

history = model.fit(train_generator,
                              steps_per_epoch=120,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=40)

As you can see, overfitting starts to be an issue again. But our validation performance is outstanding! The model is correct 90 % of the time.

plt.style.use('bmh')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b--', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.legend()
plt.show()

As the last step, let's check the model's performance with the test set.

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(os.path.join(base_dir,'test'),
                                                    target_size=(150, 150),
                                                    batch_size=25,
                                                    class_mode='binary')

model.evaluate(test_generator)

Our model is correct 90 % of the time!

![thumbs](./images/thumbs.jpeg)

