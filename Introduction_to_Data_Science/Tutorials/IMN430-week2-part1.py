import numpy as np

a = np.array([0, 1, 2, 3])
a

a = np.arange(10)
b = np.arange(1, 9, 2) # start, end (exlusive), step
c = np.linspace(0, 1, 6)  # start, end, num-points
d = np.linspace(0, 1, 5, endpoint=False)
e = np.diag(np.array([1, 2, 3, 4, 5]))
# Now generate some random points
f = np.random.rand(4)              # uniform in [0, 1]
g = np.random.randn(4)             # gaussian distribution

a


a.ndim

a.shape

len(a)


a = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
b = np.arange(10).reshape((5, 2)) # 5 x 2 array
b.T # take the transpose


c = np.array([1, 2, 3], dtype=float)
e = np.array([True, False, False, True])
f = np.array(['Bonjour', 'Hello', 'Hallo', 'Terve', 'Hej'])
c.dtype

a = np.array([1, 2, 3, 4])
a + 1
2**a
b = np.ones(4) + 1
a - b
a * b
a == b

import matplotlib.pyplot as plt # first import library

x = np.linspace(0, 3, 20)
y = np.linspace(0, 9, 20)
plt.plot(x, y)       # line plot
plt.plot(x, y, 'o')  # dot plot
plt.show()           # <-- shows the plot (not needed with Ipython/Spyder console)

# Create an array of int ranging from 5-15
question1 = np.arange(5,15)
question1

# Create an array containing 7 evenly spaced numbers between 0 and 23
question2 = np.linspace(0,23,7)
question2

#Numpy has several routines for generating artificial data following a 
#particular structure. Check this page for different types. And generate 
#an artificial numpy array with values between -1 and 1 that follow a 
#uniform data distribution. 
question3 = np.random.uniform(-1,1,10000)
# question3 uncomment to show values

# Visualise the array in an histogram in matplotlib.
import matplotlib.pyplot as plt # first import library
plt.hist(question3)
plt.show()

# Create two random numpy arrays with 10 elements. Find the Euclidean 
# distance between the arrays using arithmetic operators, hint: numpy 
# has a sqrt function
d1 = np.random.random_integers(1, 6, 10)
d2 = np.random.random_integers(1, 6, 10)
dist = np.sqrt((d1**2 + d2**2))
dist

# Part 2: Pandas

# pandas contains high-level data structures and manipulation tools designed to make data analysis fast and easy in Python. pandas is built on top of NumPy and makes it easy to use in NumPy-centric applications.
# A Series is a one-dimensional object similar to an array, list, or 
# column in a table. It will assign a labelled index to each item in the 
# Series. By default, each item will receive an index label from 0 to N, 
# where N is the length of the Series minus one.

# Import the Pandas Library
import pandas as pd

# create a Series with an arbitrary list
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'])
s

# specify an index to use when creating the Series
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],
              index=['A', 'Z', 'C', 'Y', 'E'])
s

# Using an index
s['A']

s[0] # Access using index location

s['Z'] # Access using label

# From dictionaries to series
d = {'Chicago': 1000, 'New York': 1300, 'Portland': 900, 'San Francisco': 1100,
     'Austin': 450, 'Boston': None}
d

cities = pd.Series(d)
cities

print (cities.isnull())

# A DataFrame is a tabular data structure comprised of rows and columns, 
# akin to a spreadsheet, database table, (or R's data.frame object if 
# you are familiar with R).
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
        'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions', 'Lions', 'Lions'],
        'wins': [11, 8, 10, 15, 11, 6, 10, 4],
        'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
data

football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'losses'])
football

print ("Access by index ver 1", football['losses'])

print ("Access by index ver 2", football.losses)

#Now plot with pandas
plt.scatter(football.losses, football.wins)
plt.show()

# modify the data
football['wins'] = 7
print (football)
