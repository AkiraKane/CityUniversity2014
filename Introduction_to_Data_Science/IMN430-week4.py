# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:58:30 2014

@author: Daniel Dixey
"""

# Week 04 - Investigate relations & structures

# Data: Crime vs. Socio-economic indicators

# For this week's exercises, we will be analysing a data set made available by 
# the UCI Machine Learning Repository, which is a good resource to find example 
# data sets. The data, Communities in the US., combines socio-economic data 
# from the '90 Census, law enforcement data from the 1990 Law Enforcement 
# Management and Admin Stats survey, and crime data from the 1995 FBI UCR. We 
# provide two versions of the data, a problematic one with some missing values, 
# and a clean version where the problematic columns have been removed. Detailed 
# information on column names can be found here. The reason this data set has 
# been collected is to find the relations between crime statistics and 
# socio-economic variables. Before you continue with practicals:

# Have a look at the meta-data (for the problematic version).
# Identify the dependent and independent variables in the data.

# Import Statistical Library
from scipy import stats
# Import Pandas and Numpy for Array Manipulation
import numpy as np
import pandas as pd
# Import Plotting Library
import matplotlib.pyplot as plt
# Import Statsmodels
import statsmodels.api as sm

# Load the Communities in the US (cleaned version) into a pandas data frame.
passenger = pd.read_csv('http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/censusCrimeClean.csv')

# Copy one of the dependent and one of the independent columns into separate 
# numpy arrays. You can use as_matrix function in pandas. You can choose any 
# but an interesting one could be to look at the relation between "medIncome" 
# and "ViolentCrimesPerPop". Remember to use numpy slicing you learnt earlier.

Subset = passenger.as_matrix(['ViolentCrimesPerPop','medIncome'])

# Perform a Pearson correlation and note the correlation value.
Pearson_Subset = stats.pearsonr(Subset[:,0], Subset[:,1])
print 'Pearson Correlation Coefficient: %.4f' % Pearson_Subset[0]

# Perform a Spearman correlation computation and note the correlation value.
Spearman_Subset = stats.spearmanr(Subset)
print 'Spearman Correlation Coefficient: %.4f' % Spearman_Subset[0]

# Comment on the differences / similarities in relation to a scatterplot 
# visualisation of the two columns.
plt.scatter(Subset[:,0], Subset[:,1])
plt.xlabel('Total Number of Violent Crimes per 100K Popuation')
plt.ylabel('Median Household Income')
plt.show()

# DIY Exercises - 2 : Regression Analysis

# Here we investigate how we can perform correlation analysis using Python , 
# scipy and statsmodels we've just installed. Alternatively, you can use statsmodels 
# in combination with Pandas. Now on to some exercises: 

# We first start with scipy. Use the basic functionality from scipy to perform 
# a simple linear regression. Get two columns from the data into numpy arrays 
# and use the scipy.stats.linregress function to perform a linear regression. 
# Comment on the results returned.

slope, intercept, r_value, p_value, std_err = stats.linregress(Subset[:,1], Subset[:,0])
print 'Slope: %.4f - Intercept: %.4f - R-Value: %.4f - P-Value: %.4f' % (slope, intercept, r_value, p_value)

# Although statsmodels can operate in coordination with pandas, we will first 
# use statsmodels with numpy arrays. Start by getting the data into numpy arrays 
# first. Basic functionality in scipy was limited and multiple regression was 
# not possible. Now select 2 independent variables and a single dependent variable. 
# Use the OLS function in statsmodels to perform a multiple regression operation 
# and comment on the results. Note that you first model and fit the model to your data: 

model = sm.OLS(Subset[:,0], sm.add_constant(Subset[:,1]))
results = model.fit()
print(results.summary())

# (Optional) It is also possible to use R-style functions when using pandas and 
# statsmodels in combination. Have a look at the resource here and perform the 
# same analysis in Step-2 using an R style formula.

from statsmodels.formula.api import ols

Model = ols(formula='ViolentCrimesPerPop ~ medIncome', data=passenger)
Output = Model.fit()
print Output.summary()

# DIY Exercises - 3 : Q-Q Plots

# We looked at Q-Q Plots in Week-03. They are great tools to visually compare 
# your distributions to known distributions. Here we use statsmodels qqplot 
# function to analyse the shape of our columns.

# Pick one of the columns from the Communities data and copy it into a 
# numpy array as before.

# RentMedian: rental housing - median rent (Census variable H32B from file STF1A) (numeric - decimal)
Subset_2 = passenger.as_matrix(['RentMedian'])
Subset_2.sort()

# Compare this selected column to a normal distribution. See this link for data 
# generation. Use the statsmodels qqplot function to generate a qqplot.
# Q-Q plot of Distributions
fig = sm.qqplot(Subset_2, dist=stats.norm, distargs=(), a=0, loc=Subset_2.mean(), scale=Subset_2.std(), fit=True, line='s', ax=None)
plt.show()

# Have a look at the slides from Week-03 (page 5 on the handouts) for different shapes.
# Visualise the column on a histogram and reflect on whether the shape you 

# Create an Array with the Same mean, std and length
Norm_Data = np.random.normal(Subset_2.mean(), Subset_2.std(), len(Subset_2))
# Plot both Plots on the same figure
plt.hist(Norm_Data)
plt.hist(Subset_2)
plt.show()
# inferred from Q-Q plots and the shape of the histogram correlate.






    








