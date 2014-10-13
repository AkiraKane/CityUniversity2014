# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:59:34 2014

@author: Daniel Dixey
"""

# Introduction
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],
                  columns=['one', 'two', 'three'])
df['four'] = 'bar'
df['five'] = df['one'] > 0
# Here we are generating some missing values artificially 
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
# use this function to get an overview of the nul values along a column
pd.isnull(df2['one'])

##############################
# DIY Exercises - 1 : Missing values
##############################

# 1 - Load the slightly modified Titanic survival data into a pandas data frame.
# Load directly from URL
Passenger_Data = pd.read_csv("http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/titanicSurvival_m.csv")

# 2 - Find the counts of missing values in each column
# Overall view of DataFrame
pd.isnull(Passenger_Data).sum()

# 3 - Compute the mean and other descriptive statistics and note these down, 
#     you can use this function
Passenger_Data.describe()

# 4 - Replace the missing values in "Age" and "Fare" columns with 0 values, 
#     and visualise in a scatterplot

# Fill NA with 0 
Passenger_Data.Age.fillna(0)
Passenger_Data.Fare.fillna(0)
# Import Plotting Library
import matplotlib.pyplot as plt
plt.scatter(Passenger_Data.Age, Passenger_Data.Fare)
plt.title('Age vs. Ticket Prices')
plt.xlabel("Passenger's Age")
plt.ylabel("Ticket Price")
plt.show()

# 5 - Replace the missing values in "Age" and "Fare" columns with the mean of 
#     each column, and visualise in a scatterplot

# Reload data to get the missing values
Passenger_Data = pd.read_csv("http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/titanicSurvival_m.csv")
# Fill NA with the means of each column
Passenger_Data.Age.fillna(Passenger_Data.Age.mean())
Passenger_Data.Fare.fillna(Passenger_Data.Fare.mean())
# Create plot of Data with new Values included
plt.scatter(Passenger_Data.Age, Passenger_Data.Fare)
plt.title('Age vs. Ticket Prices')
plt.xlabel("Passenger's Age")
plt.ylabel("Ticket Price")
plt.show()

# Reflect on the differences you see in these plots.
# DD: Rescaling required to notice the difference on the plot. ie. Remove Outliers
# DD: 177 of 891 values for Age are missing - approx 19.8% of the data for Age is missing.
# DD: 46 of 891 values for Fare are missing - approx 5.2 % missing data for Fare is missing.


#############################
# DIY Exercises - 2 : Outliers
#############################

