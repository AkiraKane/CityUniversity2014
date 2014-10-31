# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 16:21:21 2014

@author: dan
"""

# IMN 430 
# Introduction to Data Science
# Week 5
#

# DIY Exercises - 1 : Dimension reduction with PCA

# For practical reasons, you can get the data into a pandas array and get it to 
# a numpy array as discussed last week. If you put the data as it is to PCA 
# module, it will complain about the string column. So get rid of it by slicing 
# the numpy array: 

#Import Necessary Modules for Analysis
import pandas as pd
import numpy as np
# Import Data into a Dataframe Directly from Website
census = pd.read_csv('/home/dan/Spark_Files/IMN 430/censusCrimeClean.csv')
print (census.tail()) # check if everything is in place

# Convert Numerics to a matrix - Numpy object
numpyArray = census.as_matrix()
# Let's remove the first column
numpyArray = numpyArray[:,1::]

# Perform a principal component analysis on the data and project the data into 
# a 2-dimensional space first. Here, you can use scikit-learn's PCA decomposition. 
# Performing PCA in scikit-learn is a two stage process where the pca object 
# keeps all the model: 

# Import PCA functionality from sklearn module
from sklearn.decomposition import PCA
pca = PCA(census = pd.read_csv(n_components=2)
PCA_Vectors = pca.fit_transform(numpyArray)

# Import Plotting Library
import matplotlib.pyplot as plt

# Project your data to the components and visualise them on a scatterplot. 
# Can you notice any patterns?

# Fit Scatter plot and show
plt.scatter(PCA_Vectors[:,0],PCA_Vectors[:,1])
plt.show()

# Inspect the loadings on the principal components, can you make sense out of 
# them? Is it possible to interpret what the components stand for? If you find 
# it problematic, think of alternative ways to do this. Hint: You can make use 
# of the attributes of the PCA object, e.g., pca.components_

# Loadings of PCA Component Loadings are the Eigenvalues to the matrix
print (pca.components_)
# Number of Components
print (pca.n_components)

# Each Columns is represented a value in each vector - The vectors appear to be unsorted
print (pca.components_.size/len(pca.components_))

# Try to work on a subset of the dimensions, have a look at the component 
# loadings and try to interpret the relations between dimensions.

# Select Ten Columns from the Census Data and convert to a Numpy Matrix
Subset_10_DF = census.icol(np.arange(4,14))
Subset_10_Array = Subset_10_DF.as_matrix()

# Create the Principal Component Vector
# Define PCA model - Create One new variable
pca_s_10 = PCA(n_components=1)
# Transform the data
PCA_Subset = pca_s_10.fit_transform(Subset_10_Array)

# Print Output using a for loop and print statement
i=0
for col in Subset_10_DF:
    print 'Column: %s   - Relevancy in PCA Variable: %f.3' % (col, PCA_Subset[i])
    i=i+1

# DIY Exercises - 2 : Multidimensional scaling

# Load the data from here and select the numerical columns that can be used in 
# MDS calculations and copy them into a numpy matrix. Alternatively, you can select 
# the columns in Excel and load afterwards. Remember to load the borough names 
# as labels (we'll need them later).

London_Data = pd.read_csv('http://staff.city.ac.uk/~sbbk529/Teaching/Resources/INM430/Week05/london-borough-profiles.csv', skiprows=1)
Numpy_Array = (London_Data[['Number of active businesses, 2012','Two-year business survival rates 2012']]).as_matrix()
