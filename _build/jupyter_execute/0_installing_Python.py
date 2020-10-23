## Installing Python

There are two alternatives for installing Python:
 - Directly from installation files that can be downloaded from [www.python.org](https://www.python.org)
 - Anaconda environment that can be downloaded from [www.anaconda.com](https://www.anaconda.com)

Both options have pros and cons. Personally, I started with Anaconda, but have since moved to use "pure" Python. With Anaconda, you get much more, but also many unneeded features that waste space. Also, with pure Python, you get more recent versions of libraries using Pip. The Conda versions are always older.

### Anaconda

![snake](./images/snake.png)

The core Python package is pretty easy to install, especially in Linux. However, I would still suggest that you start your Python-journey with Anaconda [www.anaconda.com](https://www.anaconda.com/). You need many additional libraries not provided by the core installation to go through these lectures. Of course, you could manually install those libraries, but there is always a danger of compatibility issues with the most recent versions of the libraries.

So that you can entirely focus on learning Python, it is essential that everything is working. The Python libraries provided by the Anaconda environment are checked to be compatible with each other, and all are very easy to install. For example, installing GPU-enabled Tensorflow in Anaconda is just a few mouse clicks. In pure Python, it is extremely difficult, because you need to manually install the exact correct versions of the NVIDIA GPU libraries. Anaconda is also very popular and works exactly in the same way on all platforms. Also, the libraries list is quite extensive, and it is rare that a Python library is missing from the Conda environment (although this can happen).

### Installing Anaconda

To install Anaconda, you just need to download the binaries from [www.anaconda.com/products/individual](https://www.anaconda.com/products/individual) and follow the instructions. Remember to install the latest version. During the installation, you are asked whether you'd like to make Anaconda your default Python environment. You should definitely answer yes to avoid difficulties later. (Notice that if you plan to install also pure Python, then you do not necessarily want to make Anaconda your default environment. However, if you are learning Python, it is **strongly** discouraged to hassle with many Python installations. It is like asking for trouble.) 

### Updating Anaconda

Anaconda uses Conda ([conda.io](https://docs.conda.io/en/latest/)) to manage libraries and packages. You do not necessarily need to learn Conda commands, as you can do (almost) everything in Anaconda just by pointing and clicking. However, one command you should execute regularly is **conda update anaconda**. It keeps your Anaconda environment updated.

For more information on Conda, go to [docs.conda.io](https://docs.conda.io/en/latest/).

### Installing pure Python

![python](./images/python.jpg)

If you still, despite my warnings, want to install pure Python, then follow these instructions. (Just kidding :). There are benefits of using pure Python, as I mentioned earlier.)

For windows, follow these steps: [docs.python.org/3/using/windows.html#installation-steps](https://docs.python.org/3/using/windows.html#installation-steps)

Python comes preinstalled on most Linux distributions and is available as a package on all others.
If this is not the case for your Linux, follow these steps: [docs.python.org/3/using/unix.html#on-linux](https://docs.python.org/3/using/unix.html#on-linux)

### Environments

Whether you use Anaconda or pure Python, you should definitely learn to use environments. This way, you can install only the libraries needed for the current task, and other unnecessary libraries are not breaking the environment.

The instructions to create and activate environments in pure Python are here: [docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html) 

And for Anaconda, here: [docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/)

If you want to use Conda at the terminal, follow these instructions: [docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Jupyter notebook

Jupyter notebooks are one of the many possible ways to interact with Python and its scientific libraries.

They use a browser-based interface to Python with
* The ability to write and execute Python commands one command at a time.
* Observe output in a formatted form, including tables, figures, animation, etc.
* Add formatted text (Markdown) between the Python commands. It also supports mathematical expressions with MathJax.

Jupyter is nowadays an extremely popular environment for data science and scientific computing. Almost always, data science examples that are available on the Internet are in the form of Jupyter notebooks.

While Jupyter isn't the only way to code in Python, it's great for
* beginners
* quick testing
* sharing code

By the way, this book is written with Jupyter notebooks.

### Starting Jupyter notebook

Once you have installed Anaconda, you can start the Jupyter notebook.

Either by searching for Jupyter in your applications menu,

or opening up a terminal and typing **jupyter notebook**.

In Windows, you need to open the Anaconda command prompt for the previous line to work.

The first option will always open Jupyter automatically. On some platforms, the second option will output an address that you need to write to your browser for Jupyter the open. This is always the case if you are using Jupyter in a server environment.

There is also Jypterlab that is an advanced version of Jupyter. You can start it from the Anaconda menus or by writing **jupyter-lab**.

That's it! Now you have a working Python environment, and you can start coding.

```{note}
Here is a note!
`"

