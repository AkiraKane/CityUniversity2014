library(h2o)
library(RCurl)

# Links to Data
maleGitLink = "https://raw.githubusercontent.com/dandxy89/CityUniversity2014/master/Visual_Analytics/Autoencoder_Male_Dataset_Combined.csv"
femaleGitLink = "https://raw.githubusercontent.com/dandxy89/CityUniversity2014/master/Visual_Analytics/Autoencoder_Female_Dataset_Combined.csv"

# Download data as text
url_Male_Data  <- getURL(maleGitLink)
url_Female_Data <- getURL(femaleGitLink)

# Read data into Dataframe
male_data <- read.csv(text = url_Male_Data)
female_data <- read.csv(text = url_Female_Data)

# Start h2o with the follow settings
localH2O = h2o.init(nthreads = 4, max_mem_size = '4G')

# Get information about the cluster
h2o.clusterInfo(localH2O)

# Upload the male Dataset to h2o
male_data_h2o <- as.h2o(client = localH2O, 
                          object = male_data, 
                          key = 'male_data_h2o', 
                          header = TRUE)

# Upload the female Dataset to h2o
female_data_h2o <- as.h2o(client = localH2O, 
                          object = female_data, 
                          key = 'female_data_h2o', 
                          header = TRUE)

# Get summary information
summary(female_data_h2o)
summary(male_data_h2o)

# Get column names
colnames(female_data_h2o)
colnames(male_data_h2o)

h2o.table(male_data_h2o[,1])

# Shut down h2o
h2o.shutdown(localH2O)