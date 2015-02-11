## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Datasets Used  
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Dataset Links: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
## Using the Bank-full.csv

## Purpose of this dataset is to predict if the client will subscribe (yes/no) a term deposit (col 21).

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialise R environment
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Unhide if h2o is not install on your machine

## Set the CRAN mirror
local({r <- getOption("repos"); r["CRAN"] <- "http://cran.us.r-project.org"; options(repos = r)})

## The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

## Install Package
install.packages("h2o", repos=(c(paste(paste("http://s3.amazonaws.com/h2o-release/h2o/master/",version,sep=""),"/R",sep=""), getOption("repos"))))

## Import the h2o Library
suppressMessages(library(h2o))
suppressMessages(library(caret))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialise H2O Connection
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Run the demonstaration
#demo(h2o.glm)

## Start a h2o Cluster
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '4g', nthreads=-1)

## Check h2o cluster is functioning correctly
h2o.clusterInfo(localH2O)

# Check the current status
h2o.clusterStatus(localH2O)

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
validation_hex  <- h2o.assign(data[random >= .7 & random < .8,], "validation_set")
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
                   key = 'Deep_Learning_Model_x_1_1', # Assign a key value to Model
                   activation = "RectifierWithDropout", # or 'Tanh'
                   # validation = validation_hex, # Insert the Validation Dataset
                   classification = T, # Classification Task
                   nfolds = 10, # Number of Folds
                   #input_dropout_ratio = 0.2, # % of inputs dropout
                   #hidden_dropout_ratios = c(0.1,0.1), # % for nodes dropout
                   l2 = 0.00001, # L2 regularization value
                   hidden = c(50,50), # two layers at 30 nodes
                   epochs = 200, # max. no. of epochs
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
prediction_test = h2o.predict(model, newdata = test_hex[,c(1,2,3,4,5,6,7,8,12)])
prediction_train = h2o.predict(model, newdata = train_hex[,c(1,2,3,4,5,6,7,8,12)])


## Get the confusion Matrix - Predicted Confustion Matrix
confusion_pred = h2o.confusionMatrix(prediction_test[,1], test_hex[,17])
confusion_pred

# Get the confusion Matrix - Training Confustion Matrix
confusion_train = h2o.confusionMatrix(prediction_train[,1], train_hex[,17])
confusion_train

## =============================================================================
## Pack-up
## =============================================================================

## Check h2o cluster is functioning correctly
h2o.clusterInfo(localH2O)

## Shut down h2o server
h2o.shutdown(localH2O)

## =============================================================================
## Results
## =============================================================================

# > ## Display Variable importance
#   > model@model$varimp
# duration housing.yes default.no  loan.yes default.yes   balance job.student   loan.no job.admin. education.tertiary job.housemaid
# 1        1   0.7993869  0.7794384 0.7032358   0.6051757 0.5849366   0.5163379 0.4709486  0.4518152           0.433008     0.3944367
# marital.single marital.married job.retired       age education.primary housing.no job.unemployed job.technician job.self-employed
# 1      0.3882895       0.3873966   0.3858609 0.3807179         0.3801363  0.3686665      0.3658478      0.3596107         0.3582516
# job.entrepreneur education.secondary marital.divorced job.management job.unknown job.blue-collar education.unknown job.services
# 1        0.3505748            0.345786        0.3436916      0.3357938    0.333468        0.330276         0.3234178    0.2974606
# > 
#   > ## Show the Confusion Matrix
#   > model@model$confusion
# Predicted
# Actual      no  yes Error
# no     31960 3899 0.109
# yes     1864 2896 0.392
# Totals 33824 6795 0.142
# > 
#   > ## Make prediction with trained model (on training data for simplicity), prediction is stored in H2O cluster
#   > prediction_test = h2o.predict(model, newdata = test_hex[,c(1,2,3,4,5,6,7,8,12)])
# > prediction_train = h2o.predict(model, newdata = train_hex[,c(1,2,3,4,5,6,7,8,12)])
# > 
#   > 
#   > ## Get the confusion Matrix - Predicted Confustion Matrix
#   > confusion_pred = h2o.confusionMatrix(prediction_test[,1], test_hex[,17])
# > confusion_pred
# Predicted
# Actual     no yes Error
# no     3630 433 0.107
# yes     264 265 0.499
# Totals 3894 698 0.152
# > 
#   > # Get the confusion Matrix - Training Confustion Matrix
#   > confusion_train = h2o.confusionMatrix(prediction_train[,1], train_hex[,17])
# > confusion_train
# Predicted
# Actual      no  yes Error
# no     32063 3796 0.106
# yes     2158 2602 0.453
# Totals 34221 6398 0.147
