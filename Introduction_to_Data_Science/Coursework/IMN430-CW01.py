# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:16:13 2014

@author: Daniel Dixey [ackf415]

INM430 CW01 Scripts 
"""
# Import the CSV Module
import csv
# Import the Pandas Library for Data Analysis
import pandas as pd
# Import Numpy
import numpy as np
# Import Matplotlib
import matplotlib.pyplot as plt
# Import the Function called Scatter Matrix from the Pandas module
from pandas.tools.plotting import scatter_matrix
# Import Function to Identify entiries in a Path
from os import listdir
# Import Stats module of Scipy
from scipy import stats
# Import Statsmodels
import statsmodels.api as sm
# Import Confidence Interval Calculator
from statsmodels.sandbox.regression.predstd import wls_prediction_std

################################################
# Task 2 -- Prepare data #######################
################################################


################################################
# Importing Data

# Path where all 32 CSV Files are located.
path = '/home/dan/Desktop/IMN430-CW01/Data_CSV_Files/'

i=0
# For loop to Merge and Label all the datasets into one Master file.
for fileName in listdir(path):
    # Counter - Used to determine the Master File
    i=i+1
    # Open theCSV File in order to get the Name of the Metric
    fileData = csv.reader(open(path + fileName))
    # Retrieve the first line of the data
    first_line = fileData.next()
    # Extract the value in the First Cell - Eqivilent to A1 on the Spreadsheet
    first_line = first_line[0]
    # Remove the Name and replace the white space
    first_line = first_line.replace('Indicator Name: ','').replace(' ','_')
    #print DataFrame.icol([1,2,9]).head()
    if i==1:
        # Import the Data into a Pandas Dataframe        
        DataFrame = pd.read_csv(path + fileName, skiprows=19, na_values=np.nan)
        # For Loop to Select the Correct Columns 
        count=0; Link=0; Value_Col=0; Area_Col=0
        for name in DataFrame.columns:
            count = count+1
            if name == 'ONS Code (new)':
                Link = count
            elif name == 'Indicator value':
                Value_Col = count
            elif name.startswith('Area Name'):
                Area_Col  = count
            else:
                continue
        
        # Select only the Key for the Merge, Area Name and the Indicator value.
        Master = DataFrame.icol([Link-1,Area_Col-1,Value_Col-1])
        # Label the Column Appropriately        
        Name = Master.columns[2]
        Name =  Name.replace(' ','_') + '_' + first_line
        Master.columns = [Master.columns[0], Master.columns[1], Name]
    else:
        # Import the Data into a Pandas Dataframe
        DataFrame2 = pd.read_csv(path + fileName, skiprows=19, na_values=np.nan)
        # For Loop to Select the Correct Columns  
        count=0; Link=0; Value_Col=0
        for name in DataFrame2.columns:
            count=count+1
            if name == 'ONS Code (new)':
                Link=count
            elif name == 'Indicator value':
                Value_Col=count
            else:
                continue
        # Select only the Key for the Merge and  and the Indicator value
        Join = DataFrame2.iloc[:,[Link-1,Value_Col-1]]
        # Label the Column Appropriately
        Name = Join.columns[1]
        Name =  Name.replace(' ','_') + '_' + first_line
        Join.columns = [Join.columns[0], Name]
        # Merge onto the Master Dataframe
        Master = Master.merge(Join, how='outer', on='ONS Code (new)')

################################################
# Confirm Data has been loaded Correctly

# Isolate the Numeric Columns
Numerics_Master = Master.columns[2:]
# Isolate Area Labels
Area_Labels = Master.icol(1)

# Check the Data has been Loaded Correctly
print ('\n Inspection of the First 5 and Last 5 rows')
print (Master.head(5)) # Check if everything is in place  
print (Master.tail(5)) # Check if everything is in place 

# Get a Overview of the Properties of the DataFrame - Check that numerics are the correct type
print ('\n Inspect the Proprties of the Dataframe')
print (Master.info())

# Print the Summary Statistics of the DataFrame - Just for Anomalies
print ('\n Inspect the Summary Statistics of the Dataframe')
print (Master.describe())

# Inspection of the Skew and Kurtosis of each of the Indictor Values
print ('\n Inspect the Skewness of the Dataframe = %f')
print (Master.skew())
print ('\n Inspect the Kurtosis of the Dataframe')
print (Master.kurt())

################################################
# Dealing with Missing Data

# Further inspection - Checking for Non-null values
# If Non-null values existed then a decision would need to made in where to drop
# the observations or attempt to fill the NA values
print ('\n Inspect the Completeness of the Dataframe')
print (pd.isnull(Master).sum() > 0)
# For indicators with missing values
print (pd.isnull(Master).sum())

# From Inspection of the Data there are 5 indicators that have missing values
# 1 - Indicator_value_Acute_sexually_transmitted_infections
# 2 - Indicator_value_Incidence_of_malignant_melanoma 
# 3 - Indicator_value_Starting_breast_feeding

# Identify which areas are affected by the Null values
print (Master[Master.isnull().any(axis=1)])
# 1 - Indicator_value_Acute_sexually_transmitted_infections
# Two areas that have null values are located in Devon - index 14 and 190 ['Devon CC','West Devon CD']
# To deal with this method I have chosen to compare to the Other areas located closely to those above
# They are also values located in Devon - I have chosen to take the average of the 3 that are avaliable and replace the missing values when that value
All_Devon_Means = Master[Master['Area Name                                           '] \
                    .apply(lambda x: 'Devon' in x)].mean()
NA_Fill = All_Devon_Means[Numerics_Master == 'Indicator_value_Acute_sexually_transmitted_infections']
print ('NA value = %f' % (NA_Fill.values))
Master.Indicator_value_Acute_sexually_transmitted_infections = \
         Master.Indicator_value_Acute_sexually_transmitted_infections.fillna(NA_Fill.values)
print ('\n Re-Inspect the Completeness of the Dataframe')
print (pd.isnull(Master).sum())
# Identify which areas are affected by the Null values
print (Master[Master.isnull().any(axis=1)])


# 2 - Indicator_value_Incidence_of_malignant_melanoma 
# From inspection 'Tower Hamlets LB' has the only missing value
# Show plot of the Distribution of of Malignant Melanoma
plt.figure(1)
plt.hist(Master.Indicator_value_Incidence_of_malignant_melanoma.dropna())
plt.close(1)
# Identify where the Area 'London' is on one the distribution
London_Data = Master[Master['Area Name                                           '] \
                .apply(lambda x: 'London' in x)]
# Mean of London data
London_Data = London_Data.mean()
print 'Mean of London Data for Indicator = %f' % (9.07)
# The London average is slightly below the National Average - This is as I would expect
# Lets now compare to Boroughs in the vicinity of the Tower Hamlets.
# Identified Local Boroughs - https://en.wikipedia.org/wiki/London_boroughs
All_London_Data = Master[Master['Area Name                                           '] \
            .apply(lambda x: x in ['Southwark LB','Greenwich LB','Newham LB','Hackney LB'])]
plt.figure(2)
plt.hist(All_London_Data.Indicator_value_Incidence_of_malignant_melanoma.values)
plt.close(2)
# From inspection of the histogram - the median and Mean are almost identical
# The distribution of the four boroughs that have data is approximately therefore
# the mean of the four local boroughs, this will be used to replace the missing value
NA_Fill = All_London_Data.Indicator_value_Incidence_of_malignant_melanoma.mean()
print('Value of the NA Replacement: %f') % NA_Fill
# Fill NA value
Master.Indicator_value_Incidence_of_malignant_melanoma = \
        Master.Indicator_value_Incidence_of_malignant_melanoma.fillna(NA_Fill)
print ('\n Inspect the Completeness of the Dataframe')
print (pd.isnull(Master).sum())
# Identify which areas are affected by the Null values
print (Master[Master.isnull().any(axis=1)])

# 3 - Indicator_value_Starting_breast_feeding
# From inspection of the Area's that have missing values they are not localised
# View the distribution of the data to understand the distribution of the data.
plt.figure(3)
#Master.Indicator_value_Starting_breast_feeding.hist()
plt.hist(Master.Indicator_value_Starting_breast_feeding.dropna())
plt.close(3)
print 'Skewness of Indicator Starting Breast Feeding = %f' % (Master.Indicator_value_Starting_breast_feeding.skew())
# Show Mean of all the Data
print 'Mean of Indicator Starting Breast Feeding = %f' % (Master.Indicator_value_Starting_breast_feeding.mean())
# Replace the missing values with the mean of all the data - as the skew I have
# decided is mainly due to the possible outliers located approximately between 42 and 58 on the histogram
# Fill NA values
Master.Indicator_value_Starting_breast_feeding = Master.Indicator_value_Starting_breast_feeding \
            .fillna(Master.Indicator_value_Starting_breast_feeding.mean())
print ('\n Inspect the Completeness')
print (pd.isnull(Master).sum())
# Identify which areas are affected by the Null values
print (Master[Master.isnull().any(axis=1)])

################################################
# Normalising and Initial Visulisation

# Visually Understand the distributions of each of the Dimensations by Domain
# Correlation between variables in each category on a Scatter matrix and Tablular view

# Domain - Our communities
Our_Communities = Master[['Indicator_value_Deprivation','Indicator_value_Proportion_of_children_in_poverty' \
    ,'Indicator_value_Statutory_homelessness','Indicator_value_GCSE_achieved_(5A-C_inc._Eng_&_Maths)' \
    ,'Indicator_value_Violent_crime','Indicator_value_Long_term_unemployment']]
print Our_Communities.corr()
scatter_matrix(Our_Communities, alpha=0.2, diagonal='hist')
# From inspection - there are 4 variables that a skewed significantly
print Our_Communities.skew()
# Long Term Unemployment, Homelessness and Deprivation
# For the variables mentioned above they will be transformed using a log transformation


# Domain - Children's and young people's health
Young_Health = Master[['Indicator_value_Obese_Children_(Year_6)','Indicator_value_Smoking_in_pregnancy' \
            ,'Indicator_value_Starting_breast_feeding','Indicator_value_Alcohol-specific_hospital_stays_(under_18)' \
            ,'Indicator_value_Teenage_pregnancy_(under_18)']]
print Young_Health.corr()
print Young_Health.skew()
scatter_matrix(Young_Health, alpha=0.2, diagonal='hist')
# Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution
print ('Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution')
for col in Young_Health:
    Test_Values = stats.kstest(Young_Health[col], 'norm')
    print ('Column: %s - P-Value: %f') % (col, Test_Values[1])
# All P-values are less than 0.05 - Therefore this confirm that the data is normally distributed
# No transformation is required on the data


# Domain - Adults' health and lifestyle
Adult_Health = Master[['Indicator_value_Adults_smoking','Indicator_value_Increasing_and_higher_risk_drinking_' \
    ,'Indicator_value_Healthy_eating_adults','Indicator_value_Physically_active_adults','Indicator_value_Obese_adults']]
print Adult_Health.corr()
print Adult_Health.skew()
scatter_matrix(Adult_Health, alpha=0.2, diagonal='hist')
# From inspection the data of each of the variables is approximately normally distributed
# apart from one column - this could be due to outliers. The column Increasing_and_higher_risk_drinking
# will be left as the range of the values is narrow anyway.
# Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution
print ('Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution')
for col in Adult_Health:
    Test_Values = stats.kstest(Adult_Health[col], 'norm')
    print ('Column: %s - P-Value: %f') % (col, Test_Values[1])
# All P-values are less than 0.05 - Therefore this confirm that the data is normally distributed
# No transformation is required on the data


# Domain - Disease and poor health
Disease = Master[['Indicator_value_Incidence_of_malignant_melanoma','Indicator_value_Hospital_stays_for_self-harm' \
        ,'Indicator_value_Hospital_stays_for_alcohol_related_harm','Indicator_value_Drug_misuse' \
        ,'Indicator_value_People_diagnosed_with_diabetes','Indicator_value_New_cases_of_tuberculosis' \
        ,'Indicator_value_Acute_sexually_transmitted_infections','Indicator_value_Hip_fracture_in_65s_and_over']]
print Disease.corr()
scatter_matrix(Disease, alpha=0.2, diagonal='hist')
# From inspection of the Correlation Matrix and the scatter matrix the data apart from two
# variables looks fine and will not require any transformation to them.
# Indicator_values [New_cases_of_tuberculosis and Acute_sexually_transmitted_infections]
# are signifcantly skewed. A log transformation will reduce the affects of skewness
print ('Display skewness of the Variables')
Disease[['Indicator_value_New_cases_of_tuberculosis','Indicator_value_Acute_sexually_transmitted_infections']].skew()
# Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution
print ('Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution')
for col in Disease:
    Test_Values = stats.kstest(Disease[col], 'norm')
    print ('Column: %s - P-Value: %f') % (col, Test_Values[1])


# Domain - Life expectancy and causes of death
Life_and_Death = Master[['Indicator_value_Excess_winter_deaths','Indicator_value_Life_expectancy_male' \
            ,'Indicator_value_Life_expectancy_female','Indicator_value_Infant_deaths' \
            ,'Indicator_value_Smoking_related_deaths','Indicator_value_Early_deaths_heart_disease_and_stroke' \
            ,'Indicator_value_Early_deaths_cancer','Indicator_value_Road_injuries_and_deaths']]
print Life_and_Death.corr()
scatter_matrix(Life_and_Death, alpha=0.2, diagonal='hist')
# From inspection of the data all the histograms on the main diagonal are approximately distributed.
# Perform the Kolmogorov-Smirnov test for goodness of fit to the Normal Distribution
print ('Perform the Kolmogorov-Smirnov test for goodness of fit of the Normal Distribution')
for col in Life_and_Death:
    Test_Values = stats.kstest(Life_and_Death[col], 'norm')
    print ('Column: %s - P-Value: %f') % (col, Test_Values[1])

#################################################
# Normalising the Data

# Create a copy of the Master Dataframe to store the Normalised values in
Master_Normalised = Master

# Indicators require a transformation
Log_Transform = ['Indicator_value_New_cases_of_tuberculosis','Indicator_value_Deprivation' \
                ,'Indicator_value_Acute_sexually_transmitted_infections','Indicator_value_Long_term_unemployment' \
                ,'Indicator_value_Statutory_homelessness']
# Visulise Indicators that will transformed.
Master[Log_Transform].hist()
# Check the Skew of the Indicators
Master[Log_Transform].skew()


# For Loop to Transform Variables
for col in Master_Normalised.columns:
    if col in Log_Transform:
        c = 1.0
        Master_Normalised[col] = Master_Normalised[col].apply(lambda x: np.log(c + x))
    else:
        continue

# Check the Skewness of Dataframe after applying the transformation.
print ('Show Skew of Dataframe')
print (Master_Normalised.skew())
################################################


################################################
# Task 3 -- Perform analysis ###################
# Any code to perform the analysis #############

## Dependent variable:
# 1 - Deprivation - 'Indicator_value_Deprivation'
##Independent variables:
# 1 - Drug Misuse - 'Indicator_value_Drug_misuse'
# 2 - Acute sexually transmitted infections - 'Indicator_value_Acute_sexually_transmitted_infections'
# 3 - Obese Children (Year 6) - 'Indicator_value_Obese_Children_(Year_6)'

# Obtain a Subset of the Data.
Subset_Master_Normailsed = Master_Normalised[['Indicator_value_Deprivation','Indicator_value_Drug_misuse' \
        ,'Indicator_value_Acute_sexually_transmitted_infections','Indicator_value_Obese_Children_(Year_6)']]

Subset_Master_Normailsed_Area = Master_Normalised[['Area Name                                           ','Indicator_value_Deprivation','Indicator_value_Drug_misuse' \
        ,'Indicator_value_Acute_sexually_transmitted_infections','Indicator_value_Obese_Children_(Year_6)']]
# Get a Subset of Just the Numerics
Subset_Master_Normailsed_Numerics = Subset_Master_Normailsed.fillna(0)
Subset_Master_Normailsed_Numerics = Subset_Master_Normailsed_Numerics.as_matrix()

########## Outliers

# Plot - Historgrams of Each of the Variables to see there any Outliers in the Indicators
for j in Subset_Master_Normailsed.columns:
    plt.figure(j + ' - Histogram')
    plt.hist(Subset_Master_Normailsed[j].dropna(),bins=20)

# Each of the Variables could have Outliers - Further inspection is required.
# First create Additional Columns to idenfy rows whic are/not outliers
# Find the Mean and Standard Deviation of Vector/Column
# Evaluate each Row in Column
for j in Subset_Master_Normailsed.columns:
    Subset_Master_Normailsed['isOutlier - ' + j] = "#40A0C9"
    Mean = Subset_Master_Normailsed[j].mean()
    STDev = Subset_Master_Normailsed[j].std()
    for i, val in enumerate(Subset_Master_Normailsed[j]):
        if abs(val - Mean) > 2 * STDev:
            Subset_Master_Normailsed['isOutlier - ' + j][i] = "#D06B36"

# Plot Outliers of the Independent Indiators   
for i in np.arange(1,Subset_Master_Normailsed_Numerics.shape[1]):
    plt.figure(Subset_Master_Normailsed.columns[i] + ' - Outlier')
    plt.title(Subset_Master_Normailsed.columns[i] + ' - Outlier')
    plt.scatter(Subset_Master_Normailsed_Numerics[:,i], Subset_Master_Normailsed_Numerics[:,i], c = Subset_Master_Normailsed.icol(i+4), s=60)

for j in Subset_Master_Normailsed.columns:
    Subset_Master_Normailsed['isOutlier - ' + j] = 0
    Mean = Subset_Master_Normailsed[j].mean()
    STDev = Subset_Master_Normailsed[j].std()
    for i, val in enumerate(Subset_Master_Normailsed[j]):
        if abs(val - Mean) > 2 * STDev:
            Subset_Master_Normailsed['isOutlier - ' + j][i] = 1

Subset_Master_Normailsed_Outlier = pd.DataFrame(Subset_Master_Normailsed)
Subset_Master_Normailsed_Outlier['Areas'] = Area_Labels
Subset_Master_Normailsed_Outlier = Subset_Master_Normailsed_Outlier[(Subset_Master_Normailsed_Outlier['isOutlier - Indicator_value_Deprivation'] == 1) \
    | (Subset_Master_Normailsed_Outlier['isOutlier - Indicator_value_Drug_misuse'] == 1) \
    | (Subset_Master_Normailsed_Outlier['isOutlier - Indicator_value_Acute_sexually_transmitted_infections'] == 1) \
    | (Subset_Master_Normailsed_Outlier['isOutlier - Indicator_value_Obese_Children_(Year_6)'] == 1)]
    
Subset_Master_Normailsed_Outlier = Subset_Master_Normailsed_Outlier.icol([0,4,1,5,2,6,3,7,12])
Subset_Master_Normailsed_Outlier = pd.DataFrame(Subset_Master_Normailsed_Outlier)
print('Table of Outliers')
print(Subset_Master_Normailsed_Outlier)
print('Number of Outliers by Variable')
print(pd.DataFrame(Subset_Master_Normailsed_Outlier).icol([1,3,5,7]).sum())

########## Relationship Between Variables - Pearson and Visually  
# Perform a Pearson correlation and note the correlation value.
# Loop Through the Columns
for i in np.arange(1,Subset_Master_Normailsed_Numerics.shape[1]):
    Pearson_Subset = stats.pearsonr(Subset_Master_Normailsed_Numerics[:,0], Subset_Master_Normailsed_Numerics[:,i])
    print('Pearson Correlation Coefficient between %s and %s: %.4f') %  \
            (Subset_Master_Normailsed.columns[0],Subset_Master_Normailsed.columns[i],Pearson_Subset[0])
            
# Plot Relations 
for i in np.arange(1,Subset_Master_Normailsed_Numerics.shape[1]):
    plt.figure(Subset_Master_Normailsed.columns[i] + ' - Relationship to Dependent')
    plt.scatter(Subset_Master_Normailsed_Numerics[:,i], Subset_Master_Normailsed_Numerics[:,0], s=60)
    
# From the Output of the Perason Correlation coefficient mentioned above I 
# drop the Indicator Acute Secually Transmitted Diseases as there is a weak correlation
# between the variables
Subset_Master_Normailsed_1 = Master_Normalised[['Indicator_value_Deprivation','Indicator_value_Drug_misuse' \
        ,'Indicator_value_Obese_Children_(Year_6)']]
Subset_Master_Normailsed_1_Columns = Subset_Master_Normailsed_1.columns
Subset_Master_Normailsed_1 = Subset_Master_Normailsed_1.fillna(Subset_Master_Normailsed_1.icol(0).mean())

######### Development of Model
# Develop a General Linear Model of the the three Indicators
# Where Deprivation ~ C + Drug Misuse + Obese Chidren (Year_6)

######### Model 1
# Create an Array
Subset_Master_Normailsed_1 = Subset_Master_Normailsed_1.as_matrix()

# Define the Explanatory Variables
x = Subset_Master_Normailsed_1[:,1:]
x = sm.add_constant(x, prepend = True)

# Define the Prediction Variable
y = Subset_Master_Normailsed_1[:,0]

# Define the Model as above
model1 = sm.OLS(y, x).fit()

print(model1.params)
print('Model >> %s = %.4f constant + %f%s + %f%s') % \
    (Subset_Master_Normailsed_1_Columns[0], model1.params[0], model1.params[1], Subset_Master_Normailsed_1_Columns[1], model1.params[2], Subset_Master_Normailsed_1_Columns[2])
print(model1.summary())

# From inspection - looks like a good fitting model
# All the Constants are significant and the overall probability is very good too
# The only negative is the R Squared value which indicates that there more than
# half of the variation is not explained by the models

######### Model 2
# Create a New model including only Dri Misuse
# Deprivation ~ C + Drug Misuse 

# Define the Explanatory Variables
x = Subset_Master_Normailsed_1[:,1]
x = sm.add_constant(x, prepend = True)

# Define the Model as above
model2 = sm.OLS(y, x).fit()

# Print Output of Model
print(model2.params)
print('Model >> %s = %.4f constant + %f%s') % \
    (Subset_Master_Normailsed_1_Columns[0], model2.params[0], model2.params[1], Subset_Master_Normailsed_1_Columns[1])
print(model2.summary())

######### Model 3
# Remove Constant as P-Value is > 0.05
# Define the Explanatory Variables
x = Subset_Master_Normailsed_1[:,1]

# Define the Model as above
model3 = sm.OLS(y, x).fit()

# Print Output of Model
print(model3.params)
print('Model >> %s = %f%s') % \
    (Subset_Master_Normailsed_1_Columns[0], model3.params[0], Subset_Master_Normailsed_1_Columns[1])
print(model3.summary())

# Very good fit - All Varibales and Models are significant.
plt.figure('Pre Final Model')
plt.plot(x , y,'o', x, model3.fittedvalues, 'r--.')

######### Model 4
# Remove Zero Values as they drastically affect the model
# Define the Explanatory Variables
Subset_Master_Normailsed_1 = Subset_Master_Normailsed_1[(Subset_Master_Normailsed_1[:,0]>0)]
x = Subset_Master_Normailsed_1[:,1]
y = Subset_Master_Normailsed_1[:,0]

# Define the Model as above
model4 = sm.OLS(y, x).fit()

# Print Output of Model
print(model4.params)
print('Model >> %s = %f%s') % \
    (Subset_Master_Normailsed_1_Columns[0], model4.params[0], Subset_Master_Normailsed_1_Columns[1])
print(model4.summary())

# Very good fit - All Varibales and Models are significant.
prstd, iv_l, iv_u = wls_prediction_std(model4)
plt.figure('Final Model')
plt.plot(x, y,'o', x, model4.fittedvalues, 'r--.', x, iv_l,'g--', x, iv_u, 'g--')
plt.xlabel('Indicator - Drug Misues')
plt.ylabel('Indicator - log(Deprevation)')
plt.title('Model >> log(Deprevation) ~ Drug Misuse')

################################################
