## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Datasets Used  
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Dataset Links: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
## Using the Bank-full.csv

## Purpose of this dataset is to predict if the client will subscribe (yes/no) a term deposit (col 21).

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialise R environment
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Import the h2o Library
suppressMessages(library(h2o))
suppressMessages(library(caret))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialise H2O Connection
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Start a h2o Cluster
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '4g', nthreads=-1, )

## Check h2o cluster is functioning correctly
h2o.clusterInfo(localH2O)

## =============================================================================
## Import Data
## =============================================================================

## Importing data into h2o
file_location <- '/home/dan/Documents/Dropbox/Data Science Share/City - MSc Data Science/IMN427_Neural_Computing/Week2/Pole Balancing Exercise/bank-full.csv'

## Read into Data Frame
bank_data <- read.csv(file = file_location, header = TRUE, sep = ";")

## Check Structure of Data
str(bank_data)

## Upload Dataframe to h2o
data <- as.h2o(localH2O, bank_data, key = 'data')

## =============================================================================
## Build Deep Learning Network
## =============================================================================

## Get some help on the deep learning functionality in h2o
help(h2o.deeplearning)

## Split the data into train/validation/test splits of (70/10/20)
random <- h2o.runif(data, seed = 123456789)

## Create 3 Sets of Data
train_hex <- h2o.assign(data[random < .9,], "train_set")
#validation_hex  <- h2o.assign(data[random >= .7 & random < .8,], "validation_set")
test_hex  <- h2o.assign(data[random >= .9,], "test_set")

# Save as Dataframes
y_train <- as.factor(as.matrix(train_hex$y))
y_test <- as.factor(as.matrix(test_hex$y))

## Check Files have been Created Accurately
nrow(train_hex)
nrow(validation_hex)
nrow(test_hex)

## Clean up Temporaries
h2o.rm(localH2O, grep(pattern = "Last.value", x = h2o.ls(localH2O)$Key, value = TRUE))

## Build the Model
model <- 
  h2o.deeplearning(x = c(1,2,3,4,5,6,7,8,12),  # column numbers for predictors
                   y = 17,   # column number for label
                   data = train_hex, # data in H2O format
                   key = 'Deep_Learning_Model_x_1', # Assign a key value to Model
                   activation = "Rectifier", # or 'Tanh'
                   classification = T, # Classification Task
                   nfolds = 0, # Number of Folds
                   #input_dropout_ratio = 0.2, # % of inputs dropout
                   #hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(150,150,100,50), # two layers at 30 nodes
                   epochs = 300, # max. no. of epochs
                   variable_importances=T, # Get Variable Importance
                   train_samples_per_iteration = -1) 

## =============================================================================
## Evaluate Performance
## =============================================================================

## Get an Overview of the Model
summary(model)

## Display Variable importance
model@model$varimp

## Show the Confusion Matrix
model@model$confusion

## Make prediction with trained model (on training data for simplicity), prediction is stored in H2O cluster
prediction = h2o.predict(model, newdata = test_hex)

## Save results in Dataframe
res <- data.frame(Training = NA, Test = NA)

## Evaluate performance
yhat_train <- h2o.predict(model, train_hex)$predict
yhat_train <- as.factor(as.matrix(yhat_train))
yhat_test <- h2o.predict(model, test_hex)$predict
yhat_test <- as.factor(as.matrix(yhat_test))

## Print Confusion Matrix
confusionMatrix(yhat_train, y_train)$table
confusionMatrix(yhat_test, y_test)$table

## Save Results into Dataframe
res[1, 1] <- round(confusionMatrix(yhat_train, y_train)$overall[1], 4) # Save Accuracy for Training Set
res[1, 2] <- round(confusionMatrix(yhat_test, y_test)$overall[1], 4) # Save Accuracy for Test Set

## Print Results
res

## =============================================================================
## Pack-up and Time for Bed
## =============================================================================

## Check h2o cluster is functioning correctly
h2o.clusterInfo(localH2O)

## Shut down h2o server
h2o.shutdown(localH2O)

## =============================================================================
## Results
## =============================================================================

# > model
# IP Address: localhost 
# Port      : 54321 
# Parsed Data Key: train_set 
# 
# Deep Learning Model Key: Deep_Learning_Model_x_1
# 
# Training classification error: 0.02326287
# 
# Validation classification error: 1
# 
# Confusion matrix:
#   Reported on train_set 
# Predicted
# Actual     no  yes Error
# no     4802  157 0.032
# yes      73 4855 0.015
# Totals 4875 5012 0.023
# 
# Relative Variable Importance:
#   duration balance       age job.technician education.tertiary job.admin. marital.divorced job.management education.primary job.services
# 1        1 0.88573 0.6125771       0.469664          0.4685727  0.4657619        0.4529966      0.4411596         0.4399247    0.4314883
# job.retired job.self-employed housing.yes  loan.yes marital.single education.unknown education.secondary job.blue-collar housing.no
# 1    0.431256         0.4268847    0.420892 0.4200836      0.4167744         0.4148537           0.4107296       0.4064752  0.4045341
# marital.married job.housemaid job.unemployed job.student job.entrepreneur default.no   loan.no job.unknown default.yes
# 1        0.399872     0.3956018       0.385637   0.3685814        0.3609788  0.3564793 0.3514062   0.2936039   0.2727928
# 
# AUC =  0.9968182 (on train)

# Training   Test
# 1   0.9793 0.8443

# confusionMatrix(yhat_train, y_train)$table
# Reference
# Prediction    no   yes
# no  35235   216
# yes   624  4544
# > confusionMatrix(yhat_test, y_test)$table
# Reference
# Prediction   no  yes
# no  3684  336
# yes  379  193
