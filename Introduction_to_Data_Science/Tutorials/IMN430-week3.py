# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:59:34 2014

@author: Daniel Dixey
"""

# Introduction
import numpy as np
import pandas as pd
# Import Plotting Library
import matplotlib.pyplot as plt

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
Passenger_Data = pd.read_csv(
    "http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/titanicSurvival_m.csv")

# 2 - Find the counts of missing values in each column
# Overall view of DataFrame
pd.isnull(Passenger_Data).sum()

# 3 - Compute the mean and other descriptive statistics and note these down,
#     you can use this function
Passenger_Data.describe()

# 4 - Replace the missing values in "Age" and "Fare" columns with 0 values,
#     and visualise in a scatterplot

# Fill NA with 0
Passenger_Data.Age = Passenger_Data.Age.fillna(0)
Passenger_Data.Fare = Passenger_Data.Fare.fillna(0)

plt.scatter(Passenger_Data.Age, Passenger_Data.Fare)
plt.title('Age vs. Ticket Prices')
plt.xlabel("Passenger's Age")
plt.ylabel("Ticket Price")
plt.show()

# 5 - Replace the missing values in "Age" and "Fare" columns with the mean of
#     each column, and visualise in a scatterplot

# Reload data to get the missing values
Passenger_Data1 = pd.read_csv(
    "http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/titanicSurvival_m.csv")
# Fill NA with the means of each column
Passenger_Data1.Age = Passenger_Data1.Age.fillna(Passenger_Data1.Age.mean())
Passenger_Data1.Fare = Passenger_Data1.Fare.fillna(Passenger_Data1.Fare.mean())
# Create plot of Data with new Values included
plt.scatter(Passenger_Data1.Age, Passenger_Data1.Fare)
plt.title('Age vs. Ticket Prices')
plt.xlabel("Passenger's Age")
plt.ylabel("Ticket Price")
plt.show()

# Dual Plot
plt.scatter(Passenger_Data1.Age, Passenger_Data1.Fare)
plt.scatter(Passenger_Data.Age, Passenger_Data.Fare)
plt.show()

# Reflect on the differences you see in these plots.
# DD: Rescaling required to notice the difference on the plot. ie. Remove Outliers
# DD: 177 of 891 values for Age are missing - approx 19.8% of the data for Age is missing.
# DD: 46 of 891 values for Fare are missing - approx 5.2 % missing data for Fare is missing.
# DD: Outliers distort the graphiimport matplotlib.pyplot as plt.


#############################
# DIY Exercises - 2 : Outliers
#############################

# Here we look at how we can identify outliers in a dataset and observe how
# things change with outlier removal.

# 1 - Load the data on properties of cars into a pd dataframe
Car_Data = pd.read_csv(
    'http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/accord_sedan.csv')

# 2 - Visualise the columns: "price" and "mileage"
plt.scatter(Car_Data.price, Car_Data.mileage)
plt.title('Value of Car vs. Mileage')
plt.xlabel("Value")
plt.ylabel("Mileage")
plt.show()

# 3 - Identify the 2D outliers using the visualisation
# DD: Negative correlation with a main cluster quite high in density
# DD: Majority of outliers greater than 16600

# 4 - Add two new columns to the dataframe called isOutlierPrice, isOutlierAge.
#     For the Price column, calculate the mean and standard deviation. Find any
#     rows that are more than 2 times standard deviations away from the mean
# and mark them with a 1 in the isOutlierPrice column. Do the same for Age
# column
Car_Data['isOutlierPrice'] = (
    Car_Data.price > (
        Car_Data.price.mean() +
        2 *
        Car_Data.price.std())) | (
            Car_Data.price < (
                Car_Data.price.mean() -
                2 *
                Car_Data.price.std()))
Car_Data['isOutlierMileage'] = (
    Car_Data.mileage > (
        Car_Data.mileage.mean() +
        2 *
        Car_Data.mileage.std())) | (
            Car_Data.mileage < (
                Car_Data.mileage.mean() -
                2 *
                Car_Data.mileage.std()))
Car_Data['isOutlierPriceOrMilage'] = (
    Car_Data.isOutlierPrice == True) | (
        Car_Data.isOutlierMileage == True)

# Plotting Mileage Outliers
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierMileage == True)],
    Car_Data.mileage[
        (Car_Data.isOutlierMileage == True)],
    c='green',
    label='Outlier Mileage',
    s=100)
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierMileage == False)],
    Car_Data.mileage[
        (Car_Data.isOutlierMileage == False)],
    c='blue',
    label='Non-Outlier Mileage',
    s=100)
plt.title('Value of Car vs. Mileage')
plt.xlabel("Value")
plt.ylabel("Mileage")
plt.legend(loc='upper right', shadow=True)
plt.show()

# Plotting Price Outliers
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierPrice == True)],
    Car_Data.mileage[
        (Car_Data.isOutlierPrice == True)],
    c='green',
    label='Outlier Price',
    s=100)
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierPrice == False)],
    Car_Data.mileage[
        (Car_Data.isOutlierPrice == False)],
    c='blue',
    label='Non-Outlier Price',
    s=100)
plt.title('Value of Car vs. Mileage')
plt.xlabel("Value")
plt.ylabel("Mileage")
plt.legend(loc='upper right', shadow=True)
plt.show()

# Plotting Price Outliers and Milage Outliers
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierPriceOrMilage == True)],
    Car_Data.mileage[
        (Car_Data.isOutlierPriceOrMilage == True)],
    c='green',
    label='Outlier',
    s=100)
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierPriceOrMilage == False)],
    Car_Data.mileage[
        (Car_Data.isOutlierPriceOrMilage == False)],
    c='blue',
    label='Non-Outlier',
    s=100)
plt.title('Value of Car vs. Mileage')
plt.xlabel("Value")
plt.ylabel("Mileage")
plt.legend(loc='upper right', shadow=True)
plt.show()

# 5 - Visualise these values with a different color in the plot. Observe
#     whether they are the same as you would mark them.

# DD: More or less

# 6 - (Optional -- identify 2D outliers) Compute a 2D Mahalanobis distance for
# each row (you can use a scipy function). For this, you need to find the 2D
# mean vector and find the 2D Mahalanobis distance of each point to this mean
# vector. Finally, color all the points according to their mahalanobis score.
# Here is a matplotlib example that uses coloring and choose an appropriate
# color map here, for instance, Greens is a good choice. And compare your
# observations in step-3 to the resulting scatterplot.

from scipy.spatial.distance import cdist

Matrix = np.matrix([Car_Data.mileage, Car_Data.price]).transpose()
meanVector = np.matrix([Car_Data.mileage.mean(), Car_Data.price.mean()])
Car_Data['mahalanobisDistance'] = cdist(Matrix, meanVector, 'mahalanobis')

# Histogram of mahalanobisDistance
plt.hist(Car_Data.mahalanobisDistance, bins=50, color='g')

# Scatter Plot
Car_Data['isOutlierMahalanobisOutlier'] = (
    Car_Data.mahalanobisDistance > (
        Car_Data.mahalanobisDistance.mean() +
        2 *
        Car_Data.mahalanobisDistance.std())) | (
            Car_Data.mahalanobisDistance < (
                Car_Data.mahalanobisDistance.mean() -
                2 *
                Car_Data.mahalanobisDistance.std()))
# Plotting Price Outliers and Milage Outliers
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierMahalanobisOutlier == True)],
    Car_Data.mileage[
        (Car_Data.isOutlierMahalanobisOutlier == True)],
    c='green',
    label='Outlier',
    s=100)
plt.scatter(
    Car_Data.price[
        (Car_Data.isOutlierMahalanobisOutlier == False)],
    Car_Data.mileage[
        (Car_Data.isOutlierMahalanobisOutlier == False)],
    c='blue',
    label='Non-Outlier',
    s=100)
plt.title('Value of Car vs. Mileage')
plt.xlabel("Value")
plt.ylabel("Mileage")
plt.legend(loc='upper right', shadow=True)
plt.show()

#############################
# DIY Exercises - 3 : Data Transformations
#############################
# Here we test a couple of data distributions and observe how they change the data.
# 1 - Download the csv data file from WHO on Tuberculosis (from Week01).
# Information on the data can be found on WHO's web page.
Tuberculosis = pd.read_csv(
    'http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/TB_burden_countries_2014-09-29.csv')
# 2- You may need to replace missing values before you start.
pd.isnull(Tuberculosis).sum()  # Lots of missing Data!
Tuberculosis.describe()
# Drop NA values temporarily for Plotting before deciding how to replace
# the NA's
Tuberculosis1 = Tuberculosis.dropna()
Tuberculosis1.hist(xlabelsize=0.5)
# 3 - Choose a number of columns with different shapes, for instance, "e_prev_100k_hi" is left skewed and visualise on an histogram
# Chosen columns to plot: c_cdr_lo, e_inc_100k_hi, year, c_cdr, e_prev_100k_hi
Tuberculosis[['c_cdr_lo', 'e_inc_100k_hi', 'year',
              'c_cdr', 'e_prev_100k_hi']].hist(xlabelsize=0.5)
# 4 - Apply a log transformation on the data. Numpy has a log function.
# and visualise. Observe the changes
Logged_Values = np.log(
    Tuberculosis[['c_cdr_lo', 'e_inc_100k_hi', 'year', 'c_cdr', 'e_prev_100k_hi']])
# Change Columns Names
Logged_Text = '_Logged'
Logged_Values.columns = Logged_Values.columns + Logged_Text
Logged_Values.hist()
# 5 - Choose the numerical columns and map all the columns to [0,1] interval
# Identify all the Columns with Numerical Values
Tuberculosis.dtypes == 'float64'
Normalised = Tuberculosis.loc[:, Tuberculosis.dtypes == 'float64']


def normalise_func(x, max_c, min_c):
return (x - min_c) / (max_c - min_c)
for col in Normalised:
    # Max
max_c = Normalised[col].max()
# Min
min_c = Normalised[col].min()
# Create a Map Function
Normalised[col] = Normalised[col].map(
    lambda x: normalise_func(x, max_c, min_c))
# 6 - Now you can compare the means of each column.
Normalised.mean()
from pandas.tools.plotting import scatter_matrix
scatter_matrix(Normalised, alpha=0.2, figsize=(
    6, 6), diagonal='kde')  # Correlation Matrix
