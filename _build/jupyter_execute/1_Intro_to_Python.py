## Introduction to Python

### What's Python
Guido van Rossum created Python in 1989. It is a general-purpose language used in many fields, like web development, game development, multimedia, data processing, and recently, machine learning. Python is an open-source software, with development coordinated by Python Software Foundation ([www.python.org/psf/](https://www.python.org/psf/). After a slow start, the popularity of Python has exploded in the last decade and is now one of the most popular programming languages.

Python is a very intuitive and beginner-friendly programming language. However, it is not the easiest language to grasp for programming veterans that have accustomed to languages like C and Fortran. Due to its beginner-friendliness, it is often used to teach programming and has been widely adopted by the data science community. It is also replacing traditional spreadsheet applications, like Excel, mostly due to its excellent data-handling library Pandas ([pandas.pydata.org](https://pandas.pydata.org/))

Python is a high-level, interpreted language. The core of the language is quite small, but it is supported by a huge collection of libraries. It supports main programming styles but was designed as an object-oriented programming language.

### Syntax and design
Python's syntax is one of its strongest points. It is also the most controversial. After a small learning curve, the syntax is very easy to read and remember. Python has also brilliant features, like iterators, generators and list comprehension, that allow very compact coding.

Probably due to its syntax and design, it is very popular in fields, like machine learning, astronomy, meteorology, etc. This is somewhat surprising, as these fields need high computational power and Python is not a very fast language. Its popularity in finance and economics is also rising rapidly, and it is also finally finding its way to the accounting research community.

### Python basics
Python emphasises readability in its syntax. For example, **whitespaces are used to structure code**, instead of brackets. The amount of indentation separates the parts of code. For example, an if-else statement is constructed like this:

x = 3 # Just to get a meaningful response from the if-else statement
if x > 2:
    print('Larger than 2')
else:
    print('Not larger than 2')

A colon denotes the start of an indented code block. Remember that all the code must be indented by the same amount until the end of the block. Also, use the same approach for every indentation. If you use TAB for one command and SPACES with the same amount of indentation for the following command, the code will not work.

Partly due to this, Python's use of whitespaces for structuring is somewhat controversial. However, you get used to it in time. If you have blocks inside blocks, you need to indent more.

for x in range(5):
    if x > 2:
        print(str(x) + ' is larger than 2')
    else:
        print(str(x) + ' is not larger than 2')

In Python, everything is an object. All data structures, functions, classes etc. are *Python objects* with an associated type.

You can add comments to your code with the hash mark. Any text after that is ignored by Python. It is also very common to "deactivate" code by adding the hash mark in front of them. You can also add comments after code to the same line.

# Here is a comment.
print('Hello') # Another comment here.
# print('world!')

With **def**, you can define functions in Python. Here is a simple example.

def multiplier(x,y):
    return x*y

You call functions in Python with the function name and adding arguments inside brackets after the function name. With an equal sign, you can save the returned value of a function to a variable. (Almost universally, in programming languages one equal sign is reserved for setting values to variables and two equal signs **(==)** is meant for comparison, and returns a truth value.)

result = multiplier(2,4)

print(result)

As almost everything in Python is an object, they have attached functions (called methods) that can be used to access the object's internal contents. For example, strings have **lower()** that returns a lowercase version of a string.

Notice how you can execute string methods just by adding them after a string. More of this later.

"ThIs Is A pYtHoN sTrInG.".lower()

It is important to remember that with a single equal sign, you are passing references to Python objects. The object itself is not copied in the process. Here is an example. First, we create a four-element list called **a**. Then we assign a to a new variable **b**.

a = ['one','two','three','four']
b = a

Then we add value to the list where x is pointing.

a.append('five')

a

As you can see the value 5 is added to the list, where x is pointing. But it is also added to the list where y is pointing. So they are both pointing to the same list. It is important to remember that the same principle is also true for arguments passed to a function.

b

Objects have attributes and methods, and they are accessed passing a dot and the name of an attribute/method after the object's name. For example, Python lists have the following methods:
* append() - Adds an element at the end of the list
* clear() - Removes all the elements from the list
* copy() - Returns a copy of the list
* count() - Returns the number of elements with the specified value
* extend() - Add the elements of a list (or any iterable), to the end of the current list
* index() - Returns the index of the first element with the specified value
* insert() - Adds an element at the specified position
* pop() - Removes the element at the specified position
* remove() - Removes the first item with the specified value
* reverse() - Reverses the order of the list
* sort() - Sorts the list

z = [10,20,30,40,50]

We can use the methods using the format **object.method** (Remember: The first index of a list is 0):

z.append(60)

z

z.index(30)

**Modules** are used in Python to access everything from a simple function to extensive libraries, like Tensorflow. Basically, they are just files containing Python code. You can create your own module by writing Python code and saving it to a file with the **.py** extension. If your filename is my_first_module.py, then you can access the module with **import my_first_module**. Remember that the module needs to be in the same directory.

The installed libraries are also modules that are at an accessible path (not necessarily in the same directory). For example, a popular NumPy library is imported with the following command.

import numpy as np

We used a keyword **np** for numpy, so now we can acces the functios of Numpy with this keyword. Here **random** is a module inside Numpy that has a function **normal()**.

np.random.normal(size=10)

You can import individual functions/attributes/modules from a library using a **from** command.

from numpy import random

random.normal(size=5)

You can use binary operators in Python.

2+3

5-2

Python uses ****** for powers.

654**2

You cannot do binary operations between strings and numbers. However, Python still accepts operations between integers and floats.

10 + '8'

10 + 8.15

You can also do comparisons easily.

5 >= 2

3 < 2

To check if two variables are referencing to the same object, we can use **is**.

x = [10,11,12]

**y** references to the same list.

y = x

**z** references to a new list that is identical to the list where x/y are referencing.

z = [10,11,12]

x and y reference to the same object.

x is y

x and z, however, are referencsing to different objects.

x is z

Although referncing to different objets, the lists where x and z are pointing, are identical.

x == z

Here is a good introduction to Python operators: [www.w3schools.com/python/python_operators.asp](https://www.w3schools.com/python/python_operators.asp)

Python has some built in scalar types for handling numerical data, strings, booleans and dates. We go through each of them.

**int** is the name for integer values

integer_value = 654

**float** is for floating point values. At default, they are 64-bit floats.

float_value = 1.234

Python will automatically change the integer divisions with remainders to floating-point values.

10/9

You can define strings with single or double quotes.

a = 'a-string'
b = "b-string"

Triple-quotes are used for multiline strings. **\n** is a newline character in Python strings.

long = """A very
long
string"""

long

With square brackets, you can **slice** many kind of Python objects, including strings. The following commands pick up the first 5 characters and the characters at indices 3,4 and 5.

a[:5]

a[2:5]

You can concatenate string by "adding" them.

a+b

String formatting is a powerful tool. Here is an example:

temperature = '{0:.3f} degrees Celsius is {1:.2f} degrees Fahrenheit'

temperature.format(25.12345,77.22221)

More information about string formatting can be found here: [docs.python.org/3/library/string.html](https://docs.python.org/3/library/string.html)

**Type casting** is an efficient method to change the type of a variable. Remember that the boolean value of zero is False and of any other number, True.

string = '1.234'

string

type(string)

to_float = float(string)

type(to_float)

to_float

Python has a module for dates and times, called **datetime**.

import datetime

d_object = datetime.date(1999,1,10)

d_object.day

d_object.month

d_object.year

There are many convenient methods for datetime-objects, but we do not start to go through them all now. Two convenient ones are the methods for converting datetime-objects to strings and vice versa.

d_object.strftime('%d / %m / %Y')

datetime.datetime.strptime('31031978', '%d%m%Y')

From this link, you can read everything related to datetime-objects, including the full list of format codes: [docs.python.org/3/library/datetime.html?highlight=datetime](https://docs.python.org/3/library/datetime.html?highlight=datetime)

Python uses similar control flow structures, like **for**, **if**, etc. found in other programming languages.

At the beginning of this chapter, we already saw an **if-else** statement. You can use **if** without **else**, or you can chain many conditions using **elif**.

def tester(x):
    if x < 0:
        print('negative')
    elif x == 0:
        print('zero')
    else:
        print('positive')

tester(-1)

tester(0)

tester(4)

**For** loops in other programmmin languages are used to iterate over a collection. In python, you can iterate over many things using **iterators**.

for i in 'Python':
    print(i)

With **continue**, you can skip the rest of the block inside a for loop, and start immediately the next round. In the following, we skip zero when going through values from -5 to 5.

for i in range(-5,5):
    if i == 0:
        continue
    print(i, end=' ')

We have been already using **range** statements in our code. It returns an iterator that yields a sequence of evenly spaced integers.

list(range(1,10))

With **break** you can exit for loop completely.

for i in range(-5,5):
    if i == 0:
        break
    print(i, end=' ')

**While** loops are very similar. The block inside a **while** loop is executed until the condition evaluates to False. Btw, **x+=1** is a short-hand for **x = x+1**.

x=-5
while x<0:
    print(x, end = ' ')
    x+=1

You can squeeze if-else statements into a single command:

def example(x):
    print('positive') if x>0 else print('negative')

example(-1)

example(2)

# EXAMPLE

### Functions

We have already used functions several times in this chapter. As with other programming languages, functions are the main tool for reusing code. A part of code, which is repeated many times in your project, can be written only once as a function. Then, instead of writing the code again, you just call the function that has this code.

Functions start with **def**, followed by the name and a colon. **Return** is used to end a function call, and to return a Python object from the function. If the end of a function is reached without a return statement, **None** is returned automatically. You can set default values for function parameters in the function declaration (the parameter **c** in the example below). Remember that the parameters with default values need to be after positional parameters.

def first_function(a, b, c=5):
    if a>5:
        return b*c
    else:
        return b+c

first_function(6,5)

first_function(4,5)

Variables defined inside a function are define to the local **namespace** of that function. That namespace (and the variables) are destroyed, when the function call ends. You can circumvent this with a **global** statement. Usually the use of global variables is discouraged (in all programmin languages).

def sample_func():
    func_a=[1,2,3,4,5]

sample_func()

func_a

def sample_func2():
    global func_b
    func_b = [1,2,3,4,5]

sample_func2()

b

a = []
def sample_func3():
    for x in range(10):
        a.append(x)

sample_func3()

a

A nice feature in Python is the possibility to return multiple values from a function. Actually, a Python function is using a tuple (more about tuples in the next chapter) to return the values. So it is still a single Python object. Here are some examples.

def kertotaulu(x):
    return 1*x, 2*x, 3*x, 4*x, 5*x, 6*x, 7*x, 8*x, 9*x

kertotaulu(2)

a,b,c,d,e,f,g,h,j = kertotaulu(2)

a

j

In Python, functions are also objects that can be, for example, passed as parameters to another function. Here is an example.

def quadratic(x):
    return x**2

def sign_change(x):
    return -x

operations = [quadratic,sign_change]

def manipulator(value, functions):
    for function in functions:
        value = function(value)
    return value

manipulator(2, operations)

manipulator(3,operations)

In Python, you can squeeze even functions to a single statement, using the **lambda** keyword. Our quadratic() function above could be written with lambda like tihs:

squeeze_quad = lambda x: x**2

squeeze_quad(2)

**Lambda** is a convenient way to pass simple functions to other functions without declaring them separately. In the following example, we pass **lambda x:-x** as a **key** -paramter to the sort() -function.

list_of_numbers= [4,9,5,11,3]

list_of_numbers.sort()

list_of_numbers

list_of_numbers= [4,9,5,11,3]

list_of_numbers.sort(key= lambda x:-x)

list_of_numbers

As I mentioned, one of the most liked feature of Python is the ability to iterate over almost everything. This is accomplished with **iterators**. For example, you can go easilty through the elements of a list.

sample_list = ['alfa','beta','gamma','delta']
for item in sample_list:
    print(item)

What happens above is that Python first creates an iterator from **sample_list**, i.e., an object that yields items one at a time when used, for example, in a for loop.

You can build your own iterator with **generators**. Generators are functions that return a sequence of objects one at a time. It pauses after each object until the next object is reguested. Generators are created by replacing **return** with **yield** in a function definition. Here is an example:

def counter(n):
    for x in range(n):
        yield x

gener = counter(10)

gener

for item in gener:
    print(item, end=' ')

Remember that if you try to use the generator again, it returns nothing, as you have already gone through the all values, and there is no new values to return.

for item in gener:
    print(item, end=' ')

You can compactly define generators using a generator experssion.

gener2 = (2**x for x in range(5))

gener2

for item in gener2:
    print(item, end = ' ')

**Try/except** statements are very convenient tools in Python. For example, you can use them to handle few exceptions in a large dataset. Or, you can use them to experiment what is wrong in your code. Here is an example.

**Float()** can be used to change, for example, numbers in a string-format to floats. However, it can not change the name of a number (here **'four**) into a float.

sample_list = ['1','2','3','four','5']

for item in sample_list:
    print(float(item))

We can avoid this problem with a **try/except** -structure. In the example below, when we encounter a problem with the fourth item, we just return the item and do not try to transform it to a float.

sample_list = ['1','2','3','four','5']

for item in sample_list:
    try:
        print(float(item))
    except:
        print(item)

### File handling

Python has a built-in function to open files. In default, it uses read-mode to open text files.

file_handle = open('quotes.txt',encoding='utf-8')

It returns an iterator that can be used to go through the lines of the text. **Rstrip** removes end-of-the-line characters, like \n.

text_list = [item.rstrip() for item in file_handle]

Our text file has 390 lines.

len(text_list)

text_list[1]

Remember that file handles are like any other iterators. Once you have used to read through a text file, you can not use it again. You need to repeat the process to read the contents again.

file_handle = open('quotes.txt',encoding='utf-8')

**Read** can be used to read a defined number of characters from a file.

file_handle.read(20)

file_handle.read(20)

**Tell** can be used to locate how much we have already read from the file.

file_handle.tell()

To release the file back to the operating system, we need to close the file.

file_handle.close()

Be careful when using a write-mode (**mode='w'**) with **open**. If the file you are opening already exits, it will be replaced with an empty file with the same name. 

Encoding problems are very common when dealing with text files in any environment, not just Python. Therefore, it is a good practice to always define the used encoding, and to stick with a specific encoding when writing text. **utf-8** is usually a good choice.

You can check the default encoding using the built-in **sys** library.

import sys

sys.getdefaultencoding()