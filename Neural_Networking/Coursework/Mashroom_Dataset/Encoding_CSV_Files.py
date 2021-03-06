# Preparing the Banking Data for Neural Computing Coursework
# Created by:        Daniel Dixey
# Date:              28/2/2015
# Last Modified       8/3/2015

#############

# Description:
# This script imports the Banking Data
# Converts the Categorical Data to Binary
# Normalises the Numerical Data
# Combines all the above into one Dataframe
# Saves the Data into a CSV so matlab can read it in.

############################

# Import Specific Modules and Libraries

# Preparation of the Data
# Managing the Data
import pandas as pd
# Transforming the Categorical Data
from sklearn.feature_extraction import DictVectorizer as DV
# Normalising the Numerical Data to [0 1] Range
from sklearn import preprocessing
# Fast Matrix operations
import numpy as np
# Import the Time module for Timing the Process
import time


def import_Data():
    # Start Clock
    start = time.time()

    # Import the Data
    #Working_DataFrame = pd.read_csv('https://raw.githubusercontent.com/EricChiang/churn/master/data/churn.csv')
    Working_DataFrame = pd.read_table(
        '/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_3_Mashroom_Dataset/Original_Data/agaricus-lepiota.data',
        sep=',',
        header=None)

    # Import Headers
    Headers = [
        'Mushroom-Classes',
        'cap-shape',
        'cap-surface',
        'cap-color',
        'bruises?',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-root',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'stalk-color-above-ring',
        'stalk-color-below-ring',
        'veil-type',
        'veil-color',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat']

    # Add Columns Headers to Dataframe
    Working_DataFrame.columns = Headers

    # Drop Columns #11 as Others have - for Consistancy and Comparisional
    # Value in doing so
    del Working_DataFrame['stalk-root']

    # Check the Data was Imported correctly
    Working_DataFrame.head(3)

    # End Stop Watch
    end = time.time()

    # Print a Message
    print('Import Complete - Time to Complete: %.4f Seconds') % (end - start)

    # Return Output
    return Working_DataFrame


def process_Data(Working_DataFrame):
     # Start Clock
    start = time.time()

    # Get Numberic Columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    Working_DataFrame_Numerics = Working_DataFrame.select_dtypes(
        include=numerics)
    numeric_col = Working_DataFrame_Numerics.columns

    # Get Categorial Columns
    Cat_col_names = Working_DataFrame.columns - \
        numeric_col - ['Mushroom-Classes']
    Working_DataFrame_Cat = Working_DataFrame[Cat_col_names]

    # Get a dictionary for the transformation
    dict_DF = Working_DataFrame_Cat.T.to_dict().values()

    # Vectorizer
    vectorizer = DV(sparse=False)

    # Transform Dataset
    Dataset_Binary = vectorizer.fit_transform(dict_DF)

    # Get the Revised Column Names
    New_Colnames = vectorizer.get_feature_names()

    # Convert Dataset Binary to a Dataframe
    Dataset_Binary_DF = pd.DataFrame(Dataset_Binary)

    # Add columns Names
    Dataset_Binary_DF.columns = New_Colnames

    # Drop the Last two Columns
    Dataset_Binary_DF = Dataset_Binary_DF.ix[:, 0:52]

    # Convert the Binary Yes No to binary values
    Transformed_Target = pd.Categorical.from_array(
        Working_DataFrame['Mushroom-Classes']).codes

    # Convert the code to a dataframe
    Transformed_Target_DF = pd.DataFrame(Transformed_Target)

    # Add the column Names - as it was lost in the transformation
    Transformed_Target_DF.columns = ['Mushroom-Classes']

    # Normalise the Numerical Data - Convert Pandas DF to Numpy Matrix
    Working_DataFrame_Numerics = Working_DataFrame_Numerics.as_matrix()

    # Define the Scaling Range
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

    # Transform Data and then recreate Pandas Array
    Working_DataFrame_Numerics = pd.DataFrame(
        minmax_scale.fit_transform(Working_DataFrame_Numerics))

    # Add Columns Names
    Working_DataFrame_Numerics.columns = numeric_col

    # Concat all the Dataframes
    Finished_DF = pd.concat(
        [Working_DataFrame_Numerics, Dataset_Binary_DF, Transformed_Target_DF], axis=1)

    # End Stop Watch
    end = time.time()

    # Print a Message
    print('Processing Data - Time to Complete: %.4f Seconds') % (end - start)

    # Return Complete Dataframe
    return Finished_DF


def saving_Data(Finished_DF):
     # Start Clock
    start = time.time()

    # Save Encoded Dataframes
    Finished_DF.to_csv(
        '/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_3_Mashroom_Dataset/Transformed.csv',
        sep=',',
        index=False)

    # End Stop Watch
    end = time.time()

    # Print a Message
    print('Saving Dataframe to CSV - Time to Complete: %.4f Seconds') % (end - start)
    print('Save File Name: Transformed.csv\n')

# Processing Algorithm
if __name__ == "__main__":
    # Start Clock
    start1 = time.time()

    # Import the Data
    Working_DataFrame = import_Data()

    # Process Data
    Finished_DF = process_Data(Working_DataFrame)

    # Saving_Data as a CSV to Read into Matlab
    saving_Data(Finished_DF)

    # End Stop Watch
    end1 = time.time()

    # Print a Message
    print('Import, Process and Saving Complete - Time to Complete: %.4f Seconds') % (end1 - start1)
