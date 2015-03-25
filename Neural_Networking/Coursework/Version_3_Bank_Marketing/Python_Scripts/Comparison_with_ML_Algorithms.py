from __future__ import print_function
#####################################################
## Preparing the Banking Data for Neural Computing Coursework
# Created by:        Daniel Dixey
# Date:              28/2/2015
# Last Modified      21/3/2015
# No. Lines          696
#####################################################

#####################################################
### Description of Algorithm:

# 1 - This script imports the Banking Data
# 2 - Converts the Categorical Data to Binary
# 3 - Normalises the Numerical Data
# 4 - Combines all the above into one Dataframe
# 5 - Splits the data in two Parts - Training and Testing
#       where: Testing is Unseen
# Test 1: Accuracy of various Machine Learning Models
# Test 2: Accuracy, F1 Score, Log Loss of an Unboosted Neual Network
# Test 3: Accuracy, F1 Score, Log Loss of a boosted Neual Network - Using SMOTE
# Test 4: Accuracy, F1 Score, Log Loss of a boosted Neual Network - Using a Restricted Boltzmann Machine

# TO DO: Add Momentum and Regularisation into Neural Network Script

# Saves the Data into a CSV incrementally
# Also prints Statements to Console for tracking progress

# This Script utilises pre-made implementations of NN Algorithms avaliable for free online.
# Links:
# Neural Network - http://rolisz.ro/2013/04/18/neural-networks-in-python/
#                - @author: Szabo Roland <rolisz@gmail.com>

# SMOTE          - https://github.com/blacklab/nyan/blob/master/shared_modules/smote.py
#                - @author: karsten jeschkies <jeskar@web.de>

# RBM            - https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
#                - @author: Edwin Chen <hello@echen.me>

#####################################################

##### Import Specific Modules and Libraries

## Preparation of the Data
import pandas as pd                                          # Managing the Data
from sklearn.feature_extraction import DictVectorizer as DV  # Transforming the Categorical Data
from sklearn import preprocessing                            # Normalising the Numerical Data to [0 1] Range
import numpy as np                                           # Fast Matrix operations
import time                                                  # Import the Time module for Timing the Process

# Import Modules - For Machine Learning
from sklearn.cross_validation import KFold                   # Cross Validtion
from sklearn.svm import SVC                                  # Support Vector Machines
from sklearn.ensemble import RandomForestClassifier as RF    # Random Forest
from sklearn.neighbors import KNeighborsClassifier as KNN    # K-Nearest Neighbours
from sklearn.linear_model import LogisticRegression as LR    # Logistic Regression
from sklearn.ensemble import GradientBoostingClassifier as GBC # Gradient Boost Classifier
from sklearn.ensemble import AdaBoostClassifier as ABC       # AdaBoost Classifier

# Import Modules - For Neural Networking
from sklearn.grid_search import ParameterGrid                # Grid Search
from sklearn.metrics import f1_score, accuracy_score, log_loss
from random import choice
from sklearn.neighbors import NearestNeighbors

##############################################################################
### Functions to Preprocess Data
def import_Data():
    # Start Clock    
    start = time.time()    
    # Import the Data
    Working_DataFrame = pd.read_table('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_3_Bank_Marketing/Original_Data/bank-additional-full.csv', sep=';')    
    # Check the Data was Imported correctly
    Working_DataFrame.head(3)
    # Subset the Dataset to remove the nill valued durations
    Working_DataFrame = Working_DataFrame[Working_DataFrame['duration'] != 0]   
    # End Stop Watch
    end = time.time()
    # Print a Message
    print("Import Complete - Time to Complete: %.4f Seconds" % (end - start)) 
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
    print ("Pre-Processing - Time to Complete: %.4f Seconds" % (end - start))       
    # Return Complete Dataframe
    return Finished_DF

# Get Random stratified split into training set and testing sets to preserve class sizes
def stratified_Sampling(y, train_allocation=0.9):
    # Convert inputt columns to Array
    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    # Loop through Datasets to breakout into Training and Test Indicies
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_allocation*len(value_inds))
        # Identification by Boolean
        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True
    # Return Ind Vectors
    return train_inds, test_inds # stratified sampling - Returning Indicies

def saving_Data(DF, file_name):
     # Start Clock    
    start = time.time()      
    # Save Encoded Dataframes
    DF.to_csv('/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/CityUniversity2014/Neural_Networking/Coursework/Version_3_Bank_Marketing/' + file_name, sep=',', index=False)    
    # End Stop Watch
    end = time.time()    
    # Print a Message
    print("Saving Data - Time to Complete: %.4f Seconds" % (end - start))
    print('Save File Name: %s\n' % (file_name))

####################################
# Machine Learning Element

### Run the Models, Training and Collecting the Results
def run_cv(X, y, clf_class, X_Test, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    # Test the model on the Testing Dataframe
    y_test_pred =  clf.predict(X_Test)
    # Return Predicted Results
    return y_pred, y_test_pred

# Get the Accuracy of the System
def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)*100

# Create: def Classification Metrics
# Link: http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

# Running the Different Machine Learning CLassifiers
def Running_ML_Models(Training_DF, Testing_DF):
     # Start Clock    
    start = time.time()            
    # Split Data - Input Data and Target Data
    # Training Data    
    y_Train = Training_DF['y'].as_matrix()
    col = Training_DF.columns - ['y']
    X_Train = Training_DF[col].as_matrix()    
    # Testing Data
    y_Test = Testing_DF['y'].as_matrix()
    X_Test = Testing_DF[col].as_matrix()        
    # Basic Metrics of the Data
    print("Feature space holds %d observations and %d features" % (X_Train.shape[0], X_Train.shape[1]))
    print("Unique target labels:" % np.unique(y_Train))    
    # Save Results to List
    Results = []    
    
    # Testing Models Sequentially
    print ("\nTesting Models (Accuracy):\n")
    print ("Support Vector Machines:") # SVC
    Acc1, Acc2 = run_cv(X_Train, y_Train, SVC, X_Test)
    print ("Average Validation Accuracy:    %.3f" % accuracy(y_Train, Acc1))
    print ("Tesing on Unseen Dataframe:      %.3f" % accuracy(y_Test, Acc2))
    # Save Results
    Results.append(['Support Vector Machines', accuracy(y_Train, Acc1), accuracy(y_Test, Acc2)])
    
    print ("Random Forest:") #RF
    Acc1, Acc2 = run_cv(X_Train, y_Train, RF, X_Test)
    print ("Average Validation Accuracy:    %.3f" % accuracy(y_Train, Acc1))
    print ("Prediction on Testing Dataset:   %.3f" % accuracy(y_Test, Acc2))
    # Save Results
    Results.append(['Random Forest', accuracy(y_Train, Acc1), accuracy(y_Test, Acc2)])
    
    print ("K-Nearest-neighbors:") # KNN
    Acc1, Acc2 = run_cv(X_Train, y_Train, KNN, X_Test)
    print ("Average Validation Accuracy:    %.3f" % accuracy(y_Train, Acc1))
    print ("Prediction on Testing Dataset:   %.3f" % accuracy(y_Test, Acc2))
    # Save Results
    Results.append(['K-Nearest-neighbors', accuracy(y_Train, Acc1), accuracy(y_Test, Acc2)])
    
    print ("Logistic Regression:") #LR
    Acc1, Acc2 = run_cv(X_Train, y_Train, LR, X_Test)
    print ("Average Validation Accuracy:    %.3f" % accuracy(y_Train, Acc1))
    print ("Prediction on Testing Dataset:   %.3f" % accuracy(y_Test, Acc2))
    # Save Results
    Results.append(['Support Vector Machines', accuracy(y_Train, Acc1), accuracy(y_Test, Acc2)])
    
    print ("Gradient Boosting Classifier") # GBC
    Acc1, Acc2 = run_cv(X_Train, y_Train, GBC, X_Test)
    print ("Average Validation Accuracy:    %.3f" % accuracy(y_Train, Acc1))
    print ("Prediction on Testing Dataset:   %.3f" % accuracy(y_Test, Acc2))
    # Save Results
    Results.append(['Gradient Boosting Classifier', accuracy(y_Train, Acc1), accuracy(y_Test, Acc2)])
    
    print ("AdaBoost Classifier") # ABC
    Acc1, Acc2 = run_cv(X_Train, y_Train, ABC, X_Test)
    print ("Average Validation Accuracy:    %.3f" % accuracy(y_Train, Acc1))
    print ("Prediction on Testing Dataset:   %.3f" % accuracy(y_Test, Acc2))
    # Save Results
    Results.append(['AdaBoost Classifier', accuracy(y_Train, Acc1), accuracy(y_Test, Acc2)])
    
    # End Stop Watch
    end = time.time() 
    # Print a Message
    print("Testing ML Models - Time to Complete: %.4f Seconds" % (end - start))
    # Save Grid Search Values to a CSV
    saving_Data(pd.DataFrame(Results, columns=['Model', 'Validation Accuracy', 'Test Accuracy']), 'Machine_Learning_Results.csv')
    # Return Array Back to the Main Function
    return pd.DataFrame(Results, columns=['Model', 'Validation Accuracy', 'Test Accuracy'])

####################################
# Neural Networking Element of Script

# Define Activation Functions and their Respective Derivatives
def tanh(x):
    # Advised activation function taken from:
    # T.M, Heskes and B. Kappen. Online learning processes in artificial neural networks
    return 1.7159*np.tanh((2*x)/3)

def tanh_deriv(x):
    # Derivative derived from Heskes and Kappen's recommendation.
    return 1.14393*(1-(np.tanh((2*x)/3))**2)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))
    
# Define a Class that can be used for Creating and Testing the Networks
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        # Switch Function that can change the Activation function used in the Network
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # Create all the Weights in the Network
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]+ 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
    
    # Train the Neural Network
    def fit(self, X, y, learning_rate, epochs):
       # Start Clock    
       start = time.time()  
       
       X = np.atleast_2d(X)
       temp = np.ones([X.shape[0], X.shape[1]+1])
       temp[:, 0:-1] = X  # adding the bias unit to the input layer
       X = temp
       y = np.array(y)
       # Iterate through the Network - No. Epochs
       for k in range(epochs):
           # Number of Samples to Train on for each Epoch Pass
           for s in range(1):
               # Loop Through Forward and Backward with one Row
               i = np.random.randint(X.shape[0])
               a = [X[i]]
               # Forward Propagation
               for l in range(len(self.weights)):
                   a.append(self.activation(np.dot(a[l], self.weights[l])))
                   error = y[i] - a[-1]
                   deltas = [error * self.activation_deriv(a[-1])]
                   # For each Layer in the Network Backpropagate the Error
                   for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                       deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
                       deltas.reverse() # For ease of Process reverse the Array
                       for i in range(len(self.weights)):
                           layer = np.atleast_2d(a[i])
                           delta = np.atleast_2d(deltas[i])
                           # Update the Weights in the Network
                           self.weights[i] += learning_rate * layer.T.dot(delta)
       # End Stop Watch
       end = time.time() 
       # Print a Message
       print("Training Neural Network - Time to Complete: %.4f Seconds" % (end - start))
    
    # Define the Prediction Function
    def predict(self, x):
        # Convert the Test Array to a Numpy Array
         x = np.array(x)
         temp = np.ones(x.shape[0]+1)
         temp[0:-1] = x
         a = temp
         # Forward the Network
         for l in range(0, len(self.weights)):
             a = self.activation(np.dot(a, self.weights[l]))
         return a

# Testing the Models Function - Neural Network Results
def grid_search_nn(DF, Set_Name):
    # Start Clock    
    start = time.time()
    # Define the Parameters for the Grid Search
    param_grid = {'Neurons': np.arange(60,120,5), 
                  'Learning Rate': np.arange(0.05, 0.120, 0.01),
                  'epochs': np.arange(60000,120000,10000)}
    # Construct the K-Fold
    kf = KFold(len(DF), n_folds=10, shuffle=True)
    # Splitting the Data and Targets
    y_Train = DF['y'].as_matrix()
    col = DF.columns - ['y']
    X_Train = DF[col].as_matrix()
    # Storing Results After Each Run
    Averaged_Results = np.empty(shape = (len(ParameterGrid(param_grid)), 6)); In = 0
    # Grid Search through Parameters
    for parameters in ParameterGrid(param_grid):
        # Show Parameters
        #print (parameters['Neurons'], parameters['Learning Rate'], parameters['epochs'])
        # Store Results in a Array            
        Results = np.empty(shape = (10, 3)); i = 0
        # Iterate through the Cross Validation Folds
        for train_index, test_index in kf:
            # Create the Training and Test Sets
            X_train, X_val = X_Train[train_index], X_Train[test_index]
            # Get the Training and Validation Targets
            y_train, y_val = y_Train[train_index], y_Train[test_index]
            # Create the Neural Network
            nn = NeuralNetwork([X_train.shape[1], parameters['Neurons'], 1], 'tanh')
            # Train the Neural Network
            nn.fit(X_train, y_train, parameters['Learning Rate'], parameters['epochs'])
            # Prediction Using the Validation Set
            y_Pred_Val = np.zeros(shape=(X_val.shape[0], 1))
            # Validation
            for row in np.arange(0, X_val.shape[0]):
                y_Pred_Val[row] = np.round(nn.predict(X_val[row]))
            # Compute Statistics - Save: Avg Accuracy, F1 and Log Loss
            Results[i,] = [accuracy_score(y_val, y_Pred_Val), f1_score(y_val, y_Pred_Val), log_loss(y_val, y_Pred_Val)]
            # Increment Row Operator by One
            i += 1
        # Save Averaged Results - Accuracy, F1 Score, Log Loss
        Averaged_Results[In,] = [np.mean(Results[:,0]), np.mean(Results[:,1]), np.mean(Results[:,2]), parameters['Neurons'], parameters['Learning Rate'], parameters['epochs']]
        # Increment the Row Index by One
        In += 1
    # Save Grid Search Values to a CSV
    saving_Data(pd.DataFrame(Averaged_Results, columns=['Avg Accuracy','Avg F1 Score','Avg Log Loss','No. Neurons','Learning Rate','No. Epochs']), 'Grid_Search_Results_' + Set_Name + '.csv')
    # End Stop Watch
    end = time.time() 
    # Print a Message
    print("Grid Search Complete - Time to Complete: %.4f Seconds" % (end - start))
    # Return Array Back to the Main Function
    return Averaged_Results

############################
# SMOTE Boosting - http://comments.gmane.org/gmane.comp.python.scikit-learn/5278
def SMOTE(T, N, k):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """    
    n_minority_samples, n_features = T.shape
    
    if N < 100:
        #create synthetic samples only for a subset of T.
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    # Percentage of Data to Generate
    N = N/100
    
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    # Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    
    # Calculate synthetic samples
    for i in xrange(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in xrange(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]
    # Returns Synthetics Data as an Array
    return S

def get_SMOTE(Training_DF):
    # Get column Names
    colnames = Training_DF.columns
    # Splitting the Data and Targets
    y_Train = Training_DF['y'].as_matrix()
    col = Training_DF.columns - ['y']
    X_Train = Training_DF[col].as_matrix()
    # Identify the Minority Class
    Minority = X_Train[y_Train==1,]
    # Boost the Weaker Class
    Boosted = np.concatenate((SMOTE(Minority, 100, 6), np.ones((len(Minority),1))), axis=1)
    # Combine Data into a Refined Array
    SMOTE_Boosted = np.vstack((Training_DF, Boosted))
    # Return Array
    return pd.DataFrame(SMOTE_Boosted, columns=colnames)

#################################
# Restricted Boltzmann Machine Class
class RBM: 
  def __init__(self, num_visible, num_hidden, learning_rate = 0.0075):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate
    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    self.weights = 0.25 * np.random.randn(self.num_visible, self.num_hidden)    
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000):
    """
    Train the machine.
    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """
    num_examples = data.shape[0]
    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)
      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.
    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.
    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """
    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

# Getting RBM Datasets
def get_RBM_Data(Training_DF):
    # Get column Names
    colnames = Training_DF.columns
    # Splitting the Data and Targets
    y_Train = Training_DF['y'].as_matrix()
    col = Training_DF.columns - ['y']
    X_Train = Training_DF[col].as_matrix()
    # Get Weaker Class Subset
    X_Train = X_Train[y_Train==1]
    # Number of Neurons on the Hidden Layer
    N = len(X_Train)
    # Create a Restricted Boltzmann Machine
    r = RBM(num_visible = X_Train.shape[1], num_hidden = N)
    # Train the Machine
    r.train(X_Train, max_epochs = 400)
    # Get some Data from the Machine = 'Fantasies'
    Synthetic = np.identity(N)
    # Get Synthetic Data and Add Class Label onto the Data
    RBM_Data = np.concatenate((r.run_hidden(Synthetic), np.ones((len(Synthetic),1))), axis=1)
    # Combine Data into a Refined Array
    RBM_Data = np.vstack((Training_DF, RBM_Data))
    # Return Array
    return pd.DataFrame(RBM_Data, columns=colnames)

# The Main Processing Algorithm
if __name__ == "__main__":
    # Start Clock
    start1 = time.time()
    # Import the Data
    Working_DataFrame = import_Data()
    # Process Data
    Finished_DF = process_Data(Working_DataFrame)
    # Seperate Full Dataset into Training and Testing Elements    
    test_ind, training_ind = stratified_Sampling(Finished_DF['y'], train_allocation=0.1)
    # Create Training and Datasets withe     
    Testing_DF  =  Finished_DF[test_ind]
    Training_DF =  Finished_DF[training_ind]
    # Testing the Models Function - Machine Learning Results
    #Running_ML_Models(Training_DF, Testing_DF) 
    # Grid Search the Neural Network
    #Averaged_Results = grid_search_nn(Training_DF, 'Non_Boosted')
    # Get the SMOTE Boosted Adjusted Array
    #SMOTE_Boosted = get_SMOTE(Training_DF)
    # Grid Search the Neural Network - SMOTE BOOSTED
    #Averaged_Results_SMOTE = grid_search_nn(SMOTE_Boosted, 'SMOTE')   
    # Grid Search the Neural Network - RBM BOOSTED
    RBM_Boosted = get_RBM_Data(Training_DF)
    # Grid Search the Neural Network - RBM BOOSTED
    Averaged_Results_RBM = grid_search_nn(RBM_Boosted, 'RBM')   
    # Saving Testing and Training Dataframes - Testing Data frame Required for Evaulation Later
    saving_Data(Testing_DF, 'Testing.csv')
    saving_Data(Training_DF, 'Training.csv')
    #saving_Data(SMOTE_Boosted, 'SMOTE_Boosted.csv')
    saving_Data(RBM_Boosted, 'RBM_Boosted.csv')   
    # End Stop Watch
    end1 = time.time()    
    # Print a Message
    print('Import, Pre-Processing, Testing ML Models, Training 3 Neural Network Models (Non Boosted, SMOTE and RBM)\nWhere Boosting is of the Weak Class\nTime to Complete: %.4f Seconds' % (end1 - start1))