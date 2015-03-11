## Preparing the Banking Data for Neural Computing Coursework
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

## Preparation of the Data
import pandas as pd                                          # Managing the Data
from sklearn.feature_extraction import DictVectorizer as DV  # Transforming the Categorical Data
from sklearn import preprocessing                            # Normalising the Numerical Data to [0 1] Range
import numpy as np                                           # Fast Matrix operations
import time                                                  # Import the Time module for Timing the Process

# Import Modules - For Machine Learning
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC

##############################################################################
### Functions to Preprocess Data
def import_Data():
    # Start Clock    
    start = time.time()    
    
    # Import the Data
    Working_DataFrame = pd.read_table('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_3_Bank_Marketing/Original_Data/bank-additional-full.csv', sep=';')    
    
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
    Working_DataFrame_Numerics = Working_DataFrame.select_dtypes(include=numerics)
    numeric_col = Working_DataFrame_Numerics.columns

    # Get Categorial Columns
    Cat_col_names = Working_DataFrame.columns - numeric_col - ['y']
    Working_DataFrame_Cat = Working_DataFrame[Cat_col_names]
    
    # Get a dictionary for the transformation
    dict_DF = Working_DataFrame_Cat.T.to_dict().values()
    
    # Vectorizer
    vectorizer = DV( sparse = False )
    
    # Transform Dataset
    Dataset_Binary = vectorizer.fit_transform( dict_DF )
    
    # Get the Revised Column Names
    New_Colnames = vectorizer.get_feature_names()

    # Convert Dataset Binary to a Dataframe
    Dataset_Binary_DF = pd.DataFrame(Dataset_Binary)

    # Add columns Names
    Dataset_Binary_DF.columns = New_Colnames
    
    # Convert the Binary Yes No to binary values
    Transformed_Target = pd.Categorical.from_array(Working_DataFrame['y']).codes

    # Convert the code to a dataframe
    Transformed_Target_DF = pd.DataFrame(Transformed_Target)

    # Add the column Names - as it was lost in the transformation
    Transformed_Target_DF.columns = ['y']
    
    # Normalise the Numerical Data - Convert Pandas DF to Numpy Matrix
    Working_DataFrame_Numerics = Working_DataFrame_Numerics.as_matrix()

    # Define the Scaling Range
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    
    # Transform Data and then recreate Pandas Array
    Working_DataFrame_Numerics = pd.DataFrame(minmax_scale.fit_transform(Working_DataFrame_Numerics))
    
    # Add Columns Names
    Working_DataFrame_Numerics.columns = numeric_col
    
    # Concat all the Dataframes
    Finished_DF = pd.concat([Working_DataFrame_Numerics, Dataset_Binary_DF, Transformed_Target_DF], axis=1)

    # End Stop Watch
    end = time.time() 

    # Print a Message
    print('Processing Data - Time to Complete: %.4f Seconds\n') % (end - start)
       
    # Return Complete Dataframe
    return Finished_DF

### Run the Models, Training and Collecting the Results
def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    # Return Predicted Results
    return y_pred

# Get the Accuracy of the System
def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)*100

# Create: def Classification Metrics
# Link: http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

# Running the Different Machine Learning CLassifiers
def Running_Models(Finished_DF):
     # Start Clock    
    start = time.time()        
    
    # Split Data - Input Data and Target Data
    y = Finished_DF['y'].as_matrix()
    col = Finished_DF.columns - ['y']
    X = Finished_DF[col].as_matrix()
    
    # Basic Metrics of the Data
    print "Feature space holds %d observations and %d features" % X.shape
    print "Unique target labels:", np.unique(y)
    
    
    # Testing Models Sequentially
    print "\nTesting Models (Accuracy):"
    print "Support vector machines:"
    print "%.3f" % accuracy(y, run_cv(X,y,SVC))
    print "Random forest:"
    print "%.3f" % accuracy(y, run_cv(X,y,RF))
    print "K-Nearest-neighbors:"
    print "%.3f" % accuracy(y, run_cv(X,y,KNN))
    print "Logistic Regression:"
    print "%.3f" % accuracy(y, run_cv(X,y,LR))
    print "Gradient Boosting Classifier"
    print "%.3f" % accuracy(y, run_cv(X,y,GBC))
    print "AdaBoost Classifier"
    print "%.3f" % accuracy(y, run_cv(X,y,ABC))
    
    # End Stop Watch
    end = time.time() 

    # Print a Message
    print('Testing Models - Time to Complete: %.4f Seconds\n') % (end - start)
    
# The Main Processing Algorithm
if __name__ == "__main__":
    # Start Clock
    start1 = time.time()

    # Import the Data
    Working_DataFrame = import_Data()

    # Process Data
    Finished_DF = process_Data(Working_DataFrame)
    
    # Testing the Models Function
    Running_Models(Finished_DF)
    
    # End Stop Watch
    end1 = time.time()
    
    # Print a Message
    print('Import, Process and Test Multiple Models - Time to Complete: %.4f Seconds') % (end1 - start1)
    
    # End of Algorithm