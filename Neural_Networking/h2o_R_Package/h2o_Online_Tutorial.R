# Import the h2o Library
library(h2o)
# Start a h2o Cluster
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '4g', nthreads=-1, )
# Check h2o cluster is functioning correctly
h2o.clusterInfo(localH2O)

#######
# Importing data into h2o

# 1 - Importing a Dataframe for R into h2o ie from CSV Files ect...
# Import a pre-made dataset
data(iris)
# Check data is imported correctly
summary(iris)
# Convert data to r-file
iris.r <- iris
# Add file to Cluster
iris.h2o <- as.h2o(localH2O, iris.r, key="iris.h2o")

# 2 - Upload data from a URL directly
h2o.importURL(localH2O, path = 'http://www.apho.org.uk/resource/view.aspx?RID=126840',key = 'URL_File_1.hex')
# Check Files have been uploaded correctly
h2o.ls(localH2O)

# 3 - Import Local Files into h2o
h2o.importFile(localH2O, path = 'some_file_and_directory', key = 'h2o Key name')

# 4 - Import a whole directory into h2o
h2o.importFolder(localH2O, path = 'some_file_and_directory', key = 'h2o Key name')

############
# Data Manipulation

# Check if any variables in Dataset are factors
h2o.anyFactor(iris.h2o)
# Get a quick overview of the Data
summary(iris.h2o)
# Inspect the top of the Data frame
head(iris.h2o); tail(iris.h2o)
# Get the colnames from the Data
colnames(iris.h2o)
# Number of rows and columns
nrow(iris.h2o); ncol(iris.h2o)
# Get the quantiles of the Data
quantile(iris.h2o$Sepal.Length)
# Get some outlier information
iris.qs = quantile(iris.h2o$Sepal.Length, probs = c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95), na.rm = TRUE)
# Print the basic outlier detection 
print(iris.qs)
# Get the index of the Outliers
outliers.ind = iris.h2o$Sepal.Length <= iris.qs["5%"] |iris.h2o$Sepal.Length >= iris.qs["95%"]
# Create a temporary Dataframe
temp = iris.h2o[outliers.ind,]
# Add dataset to h2o
arrdelay.outliers = h2o.assign(temp, "arrdelay.outliers")
# Get some information the outliers
nrow(arrdelay.outliers)
head(arrdelay.outliers)
# Drop outliers from data
temp = iris.h2o[!outliers.ind,]
arrdelay.trim = h2o.assign(temp, "arrdelay.trim")
nrow(arrdelay.trim)

###### Building a model
# Construct test and training sets
s = runif(nrow(iris.h2o))
# Create a temporary dataframe
temp = iris.h2o[s <= 0.8,]
# Create training Set
train = h2o.assign(temp, "train")
# Get the other sample
temp = iris.h2o[s > 0.8,]
test = h2o.assign(temp, "test")
nrow(train) + nrow(test)

###### K-Means Clustering on Iris Dataset
iris_k_means = h2o.kmeans(data = iris.h2o, centers = 3, iter.max = 200,
                         cols = c("Sepal.Length", "Sepal.Width",  "Petal.Length", "Petal.Width"))
# Get details about the clustering
iris_k_means@model
# Number of iterations to find the clusters
iris_k_means@model$iter
# Create Prediction values
iris.pred = h2o.predict(object = iris_k_means, newdata = iris.h2o)



######### Deep Learning Algorithm
# Import data into H2O cluster and run Summary
train = h2o.importFile(localH2O, 
                       path = 'https://raw.githubusercontent.com/0xdata/h2o/master/smalldata/logreg/prostate.csv',  ## can be a local path to file
                       header = T,
                       sep = ',', 
                       key = 'prostate.hex')
# Get an overview of the Output
summary(train)
# Train a DeepLearning model on the H2O cluster using 3 hidden layers with 10 neurons each, Tanh activation function, 10000 epochs, predict CAPSULE from other predictors (ignore column 1: ID)
model = h2o.deeplearning(x = 3:8, y = 2, 
                         data = train, 
                         activation = "Tanh", 
                         hidden = c(10, 10, 10,10,10), 
                         epochs = 100000)
# Show the Confusion Matrix
model@model$confusion
# Make prediction with trained model (on training data for simplicity), prediction is stored in H2O cluster
prediction = h2o.predict(model, newdata = train)
# Download prediction from H2O cluster into R environment
pred = as.data.frame(prediction)
head(pred)
tail(pred)
# Check performance of binary classification model and return the probability threshold ("Best Cutoff") for optimal F1 score
#?h2o.performance
per = h2o.performance(prediction[,3], train[,2], measure = "F1")
per
# Shut down h2o server
h2o.shutdown(localH2O)
