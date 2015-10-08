__author__ = 'Daniel Dixey'

# Basic operations:

# Basic mathematical operations can be done on the console in Python.
# Python acts here like an interpreter.

1 + 2 * 3
(1 + 2) * 3

# Loops in Python:

# Loops are a common method to run a particular task several times, they
# are central in all modern programming languages. range() is a function
# that returns a series of items that you can iterate on. Try Ctrl+I on
# range to see what it does.

for i in range(1, 10):
    print(i)

# Variables in Python:

# Variables store values, they can be looked at or changed within the code.

total = 0
for i in range(1, 10):
    total = total + i
    print(total)
print ("Final Total: ", total)

# Functions in Python:

# Functions are needed to encapsulate functionalities and make them resusable.


def say_hello_to(name):
    print("Hello " + name)

say_hello_to("Cagatay")
say_hello_to("Fred")

# Conditionals in Python:

# Conditions are also central to programming to model the logical flow of
# the code. Change the value variable to see how the if-conditions behave:

value = 5
if value > 0:
    print("This is positive!")
elif value < 0:
    print("This is negative!")
else:
    print("This is 0, you can't fool me, I'm a clever piece of code")

# Comments in Python:

# Any line that starts with a # is ignored.

# Please ignore me.

# Data types in Python:

# Basic types:

x = 3          # numbers
a = "gorillas"  # strings
t = True       # booleans

# Lists:

# What if you have several of the same sort of things? You can use lists
# to store collections of things. An empty list is declared as []. And we
# can add elements with append.

myFavouriteParks = []
myFavouriteParks.append("Victoria")
myFavouriteParks.append("Hampstead Heath")
myFavouriteParks.append("Richmond")
print (myFavouriteParks)
print ("-----------------------------------")
print ("Now the same with a for-loop:")
for value in myFavouriteParks:
    print (value)

# Dictionaries

# The other main data type is the dictionary. The dictionary allows you to
# associate one piece of data (a "key") with another (a "value"). The
# analogy comes from real-life dictionaries, where we associate a word
# (the "key") with its meaning. Declare an empty dictionary with {}

foods = {}
foods["banana"] = "Yellow"
foods["avocado"] = "green"
foods
foods["banana"]
print ("now trying with a non-existent key:")
foods["cheese"]

# Loading packages in Python: As we will be making use of several packages
# and libraries in Python. We need to load them before we use them. This
# is where import comes into play.

print ("We need to load the package first:")
import numpy as np
myArray = np.arange(10)
print (myArray)

# Import Libraries

import pandas as pd

# import matplotlib.pyplot as plt

s = pd.Series([1, 3, 5, np.nan, 6, 8])

print s

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': 'foo'})

print df2
