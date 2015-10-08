##############################################################################
# Introduction to Data Science Coursework - COursework 2
# Daniel Dixey - ackf415
# 8/12/2014
# Utilizing the Book - Crossing Dataset
##############################################################################

##############################################################################
# Import Pandas Modules
import pandas as pd
# Import Numpy for fast array manipulation and mathematics
import numpy as np
# Import Plotting Libraries
import matplotlib.pyplot as plt
# Import the Time Module
import time
# Import sklearn Modules - for Logistic Regression
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
# Compute clustering with K-Means
from sklearn.cluster import KMeans
# Compute clustering with MeanShift
from sklearn.cluster import MeanShift, estimate_bandwidth
##############################################################################
# Start Time
t0 = time.time()

##############################################################################
# Import Data into Python
# Import User Data
user_data = pd.read_csv(
    '/home/dan/Desktop/IMN430-CW02/Book_Data/BX-Users.csv',
    sep=';')
# Check Data has been imported correctly
print(user_data.head(5))

# Import Ratings Data
rating_data = pd.read_csv(
    '/home/dan/Desktop/IMN430-CW02/Book_Data/BX-Book-Ratings.csv',
    sep=';')
# Check Data has been imported correctly
print(rating_data.head(5))
# Import the Book Information
book_data = pd.read_csv('/home/dan/Desktop/IMN430-CW02/Book_Data/BX-Books.csv',
                        sep=';', error_bad_lines=False)
# Check Data has been imported correctly
print(book_data.head(5))


##############################################################################
# Data Cleansing - User Data
# Check number of Commas in Cell
user_data['No_Commas'] = user_data.Location.apply(lambda x: int(x.count(',')))
# Get a Subset of the Data where No. of Commas = 2 so vectors can be made
# - [Town, State, Country]
user_data = user_data[user_data['No_Commas'].apply(lambda x: x == 2)]
# Get Town, State and Country as New Values AND remove White Space
user_data['Town'] = user_data.Location.apply(
    lambda x: x.split(',')[0].lstrip().rstrip())
user_data['State'] = user_data.Location.apply(
    lambda x: x.split(',')[1].lstrip().rstrip())
user_data['Country'] = user_data.Location.apply(
    lambda x: x.split(',')[2].lstrip().rstrip())
# Access the Number of Users where they have completed Age on their
print('Number of Null Values: %i, Number of values that will Remain: %i') % \
    (pd.isnull(user_data.Age).sum(), user_data.Age.count())
# Remove Null Age Row from the Data Set
user_data = user_data.dropna(axis=0)
# Remove Unrealistic Ages from the orginally Data Set and Recalculate
user_data = user_data[(user_data.Age <= 80) & (user_data.Age >= 15)]
##############################################################################


##############################################################################
# Merge Data into one DataFrame for Further Analysis
# Understand the Shape of Each of the Datasets prior to Merging
print('Ratings Data contains %i rows and %i Columns\n') %  \
    (rating_data.shape[0], rating_data.shape[1])
print('User Data contains %i rows and %i Columns\n') % \
    (user_data.shape[0], user_data.shape[1])
print('Book Data contains %i rows and %i Columns\n') % \
    (book_data.shape[0], book_data.shape[1])
# Merge into a Master DF - Join User Data onto Rating
DF = pd.merge(rating_data, user_data, how='left',
              left_on='User-ID', right_on='User-ID')
DF = DF.merge(book_data, how='left', on='ISBN')
# Check if there any Null values after Merge
print('Attribute Analysis shown below: \n')
DF.count() - pd.isnull(DF).sum()
# Remove Null Values from Data
DF = DF.dropna(axis=0)
# Print Shape of Data to understand the Amount of Data left in the Dataset
print('Master Dataframe has %i rows and %i columns') % (
    DF.shape[0], DF.shape[1])


##############################################################################
# Save Dataframe to CSV for Exporting to Tableau
## DF.to_csv('/home/dan/Desktop/output.csv', sep=',')


##############################################################################
# Import Country List
# Import Properly formatted Country Names - Obtained for a free database online
country_names = pd.read_csv(
    '/home/dan/Desktop/IMN430-CW02/Book_Data/countries-20140629.csv',
    sep=',')
# Check Data has been irosstmported correctly
print(country_names.head(5))
# Merge Data on to Master Dataframe - Left Join to include only those in
# the Review Data
DF = DF.merge(country_names, left_on='Country', right_on='Lower_Case_Value')
# Remove Null Values from Data to avoid Unexpected Countries
DF = DF.dropna(axis=0)
# Remove un-needed Columns
DF = DF.drop(['Country', 'Lower_Case_Value'], axis=1)
# Print Shape of Data to understand the Amount of Data left in the Dataset
print('Master Dataframe has %i rows and %i columns') % (
    DF.shape[0], DF.shape[1])

##############################################################################
# Analytical Question to Answer
# Can you use book reviews be utilised to undestand profile behavioural or demographic similarities between users?
#
# Can you use book reviews be used to understand feedback habits of the reviewers in order to enhance recommendation for all users?
#    Does the length of a book title effect the distrbution of book reviews? by each group
#             length of a book title vs average rating
#    Does the year of puplication imply popularity? by each group
#             Year of publication vs avg rating
#    Is the the spread of the distributions of user types even accross each country? by each group
#             May help to understand the types of Users of the Community
#    Finally, do particular Publishers attract certain Types of Book Reviewers?
#            Who is the more affected by this
# Compare old "Top Book" to Normalised New Value
#           Further Analysis - IF time permitted develop a Recommender System to utilize this analysis
#           python-recsys - A python library for implementing a recommender system.
#           https://github.com/ocelma/python-recsys

# Crediable Review
##############################################################################


# Plot Average Rating by the Variance to see if it is possible to identify
# any possible types of behaviour - Subset the Users and Rating
userRating = DF[['User-ID', 'Book-Rating']
                ].groupby(by='User-ID', as_index=False)
# Find the Mean and STD
userRating = userRating['Book-Rating'] \
    .agg({'Standard Deviation': np.std, 'Mean': np.mean})
# Remove the NaN Values
userRating = userRating.dropna()
# Create a Matrix for plotting
userRating_matrix = userRating.as_matrix(
    columns=['Standard Deviation', 'Mean'])
# Create plot - Define some Random Colors
colors = colors = np.random.rand(userRating.shape[0])
# Create Plot using the data from the Matrix
plt.scatter(
    userRating_matrix[
        :, 0], userRating_matrix[
            :, 1], c=colors, alpha=0.5)
# No Recognisable Cluster or Pattern
########################################
# Try K-Means to See there are Possible Patterns - Manually Define the
# Number of Clusters
k_means = KMeans(n_clusters=3)
# Create the K-Means Cluster
k_means.fit(userRating_matrix)
# Create  a New Colour Vector to Visulise the Dataset
colors = k_means.labels_
# Create Plot using the data from the Matrix
plt.figure(1, figsize=(6, 6))
plt.scatter(
    userRating_matrix[
        :, 0], userRating_matrix[
            :, 1], c=colors, alpha=0.5)
plt.xlabel('Standard Deviation')
plt.ylabel('Mean')
plt.title('Plot of Reviewers Feedback\n Using the K-Means Clustering Technique')
plt.savefig(
    '/home/dan/Desktop/IMN430-CW02/Book_Data/KMeans_Cluster_Reviews.png',
    dpi=120)
plt.show()
########################################
# Create Plot using the data from the Matrix
plt.figure(1, figsize=(6, 6))
plt.scatter(userRating_matrix[:, 0], userRating_matrix[:, 1], alpha=0.5)
plt.xlabel('Standard Deviation')
plt.ylabel('Mean')
plt.title('Plot of Reviewers Feedback')
plt.savefig('/home/dan/Desktop/IMN430-CW02/Book_Data/First_plot.png', dpi=120)
plt.show()
########################################
# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(userRating_matrix, quantile=0.2,
                               n_samples=userRating_matrix.shape[0])
# Compute the MeanShift Clusters
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(userRating_matrix)
# Define the Colurs of the Plot using the Labels
colors = ms.labels_
# Get the Label Data
labels_unique = np.unique(colors)
n_clusters_ = len(labels_unique)
# Print the Number of Estimated Clusters to the Terminal
print("Number of Estimated Clusters : %d" % n_clusters_)
plt.figure(2, figsize=(6, 6))
plt.scatter(
    userRating_matrix[
        :, 0], userRating_matrix[
            :, 1], c=colors, alpha=0.5)
plt.xlabel('Standard Deviation')
plt.ylabel('Mean')
plt.title('Plot of Reviewers Feedback\n Using the Meanshift Clustering Technique')
plt.savefig(
    '/home/dan/Desktop/IMN430-CW02/Book_Data/MeanShift_Cluster_Reviews.png',
    dpi=120)
plt.legend()
plt.show()
##############################################################################

##############################################################################
# Creating New Variables - Number of Reviews
userRatingDetails = DF[['User-ID', 'Book-Rating']]
userRatingDetails = userRatingDetails.groupby(by='User-ID', as_index=False)
# Using the Aggregate Function
userRatingDetails = userRatingDetails['Book-Rating'] \
    .agg({'USER_N_Reviews': np.size})
# Merge onto One Dataframe
DF = DF.merge(
    userRatingDetails,
    how='left',
    left_on='User-ID',
    right_on='User-ID')

##############################################################################
# Does the length of a book title effect the distrbution of book reviews?
# Using the Meanshift Clustering Technique
colors = ms.labels_
# Make Dictionary to User for Merging Later
Label_Merge = {'ID': userRating['User-ID'], 'Behaviour_Group_Code': colors,
               'Reviewer_Mean': userRating_matrix[:, 1], 'Reviewer_STD': userRating_matrix[:, 0]}
# Create DataFrame for the Merge Later
Label_Merge = pd.DataFrame(Label_Merge)
# Assign Labels to Integer Group Labels


def getName(x):
    if x == 0:
        return 'Negative Bias Reviewers'
    elif x == 1:
        return 'Positive Bias Reviewers'
    else:
        return 'Always Negative'
# Add Labels to Orginal Master Dataframe
Label_Merge['Behaviour_Group_Name'] = Label_Merge[
    'Behaviour_Group_Code'].apply(lambda x: getName(x))
# Merge Lables on to the Master DataFrame
DF = DF.merge(Label_Merge, how='left', left_on='User-ID', right_on='ID')
# Drop any Null Values
DF = DF.dropna()
# Find the Length of Books Title
DF['lenghtBookTile'] = DF['Book-Title'].apply(lambda x: len(x))
# Comparison Between Groups to understand the Differences - Raw Rating
lenComparision = pd.crosstab(rows=DF['lenghtBookTile'], cols=DF['Behaviour_Group_Name'],
                             values=DF['Book-Rating'], margins=True, aggfunc=[np.mean, np.std])
# Save file to CSV and Modify in LibreOffice
#lenComparision.to_csv('/home/dan/Desktop/lenComparision.csv', sep=',')
# Read in File into a Pandas Dataframe
lenComparision = pd.read_csv('/home/dan/Desktop/lenComparision.csv', sep=',')
# Drop Null values
lenComparision = lenComparision.dropna()
# compute one-way ANOVA P value
from scipy import stats
# Test using one-way ANCOVA for difference
f_val, p_val = stats.f_oneway(lenComparision['Always Negative_Mean'],
                              lenComparision['Negative Bias Reviewers_Mean'],
                              lenComparision['Positive Bias Reviewers_Mean'])
# Print the Results
print('One-way ANOVA P = ')
print(p_val)
# If P > 0.05, we can claim with high confidence that the means of the results
#                     of all three experiments are not significantly different.
print('Significant Evidence to Suggest that there is a difference between the Groups')

##############################################################################
# Can a Model be built such that It can determine if a user will be a
# Always negative User
DF.columns = [
    'User_ID',
    'ISBN',
    'Book_Rating',
    'Location',
    'Age',
    'No_Commas',
    'Town',
    'State',
    'Book_Title',
    'Book_Author',
    'Year_Of_Publication',
    'Publisher',
    'Image_URL_S',
    'Image_URL_M',
    'Image_URL_L',
    'Correct_Name',
    'Latitude',
    'Longitude',
    'USER_N_Reviews',
    'Behaviour_Group_Code',
    'ID',
    'Reviewer_Mean',
    'Reviewer_STD',
    'Behaviour_Group_Name',
    'lenghtBookTile']
# Create a Boolean Column to ID those in the Always Negative Group
DF['Always_Negative_Boolean'] = DF.Behaviour_Group_Code.apply(
    lambda x: 1 if x == 2 else 0)
# Add an additional Column
le = preprocessing.LabelEncoder()
# Get the Country as a integer
Country = le.fit(DF.Correct_Name.unique())
# Create a Test Case
Test_Country = le.transform(['United Kingdom'])
# Transform Data in DF
Country = le.transform(DF.Correct_Name)
# Get the Publisher as a integer
State = le.fit(DF.State.unique())
# Create a Test Case
Test_State = le.transform(['greater london'])
# Trasform the Master Dataframe
State = le.transform(DF.State)
# Create the Matrix
X = DF[['Book_Rating', 'Age', 'Reviewer_Mean', 'Reviewer_STD', 'USER_N_Reviews']]
X = X.as_matrix()
X = np.column_stack((X, Country))
X = np.column_stack((X, State))
# Get the Target Matrix
y = DF.Always_Negative_Boolean.as_matrix()
# Model = 'Always_Negative_Boolean ~ Book_Rating + Age + Reviewer_Mean + Reviewer_STD + USER_N_Reviews + \
#                                        Country + State'
# Using Cross Validation Test the Model - 10 Folds
scores_acc = cross_val_score(
    LogisticRegression(),
    X,
    y,
    scoring='accuracy',
    cv=10)
scores_f1 = cross_val_score(LogisticRegression(), X, y, scoring='f1', cv=10)
# Print the Output Scores to the Console
print('The Mean Accuracy for the 10 Folds: %.4f %%') % (100 * scores_acc.mean())
print('The Mean F1 for the 10 Folds: %.4f %%') % (100 * scores_f1.mean())
# Create Module for Self - Evaluation
model = LogisticRegression()
model = model.fit(X, y)
# Create a Test Case
# Where - My most Recent Rating is 10
#       - Age = 25
#       - Average Rating = 2 - Very Low
#       - Standard Deviation = 1 - Low Variation between Ratings
#       - Country = UK
#       - State = Greater London
Values = model.predict_proba(
    np.array([10, 25, 2, 1, 100, int(Test_Country), int(Test_State)]))
print('Prediction of User Being an Always Negative User Class: %.4f %%') % (
    100 * Values[0, 1])
# Add Predicted Results to Dataframe for Visulisation Later
DF['Predicted_Results'] = model.predict(X)
# Predicted the Number in this Group
print('The Proportion of the Populations in this Group: %.3f %%') % (100 * y.mean())
# Print Confusion Matrix - Very Good
print metrics.confusion_matrix(y, model.predict(X))

##############################################################################
# Does the year of puplication imply popularity? by each group
Year_Popularity = DF[['Behaviour_Group_Name',
                      'Year_Of_Publication', 'Book_Rating']]
# Create a Multi-Leveled Dataframe to obtain Statistics
Year_Popularity = Year_Popularity.groupby(by=['Behaviour_Group_Name', 'Year_Of_Publication'],
                                          as_index=False)
# Obtain the Mean and Standard Deviation for ploting the mean with
# confidence interval
Year_Popularity = Year_Popularity['Book_Rating'] \
    .agg({'Standard Deviation': np.std, 'Mean': np.mean,
          'No. Reviews': np.size})
# Save dataframe to CSV for Publishing in Tableau
Year_Popularity.to_csv('/home/dan/Desktop/YearPopularity.csv', sep=',')

##############################################################################
# Is the the spread of the distributions of user types even accross each
# country?
Country_Distribution = DF[['Correct_Name', 'Behaviour_Group_Name', 'User_ID']]
# Create Country Groups
Country_Distribution = Country_Distribution.groupby(by=['Correct_Name', 'Behaviour_Group_Name'],
                                                    as_index=False)
# Country Numbers by Population
Country_Distribution = Country_Distribution[
    'User_ID'].agg({'No. of Reviews by Class': np.size})
# Save dataframe to CSV for Publishing in Tableau
Country_Distribution.to_csv(
    '/home/dan/Desktop/Country_Distribution.csv', sep=',')

##############################################################################
# do particular Publishers attract certain Types of Book Reviewers?
# Subset Master Dataframe
Publisher_Overview = DF[['Publisher', 'Behaviour_Group_Name', 'Book_Rating']]
# Group Data by Publisher and Behavioural Group
Publisher_Overview = Publisher_Overview.groupby(by=['Publisher', 'Behaviour_Group_Name'],
                                                as_index=False)
# Find the Number of Reviews and average Rating
Publisher_Overview = Publisher_Overview['Book_Rating'].agg({'No. of Reviews by Class': np.size,
                                                            'Average Rating': np.mean})
# Group by Publisher to get the number of Reviews for each Publisher
Publisher_Overview_Num = Publisher_Overview.groupby(
    by=['Publisher'], as_index=False)
Publisher_Overview_Num = Publisher_Overview_Num[
    'No. of Reviews by Class'].agg({'No. Reviews': np.sum})
# Filter out low values
Publisher_Overview_Num = Publisher_Overview_Num[
    (Publisher_Overview_Num['No. Reviews'] > 5)]

# Save Master Dataframe to CSV
DF.to_csv('/home/dan/Desktop/Master.csv', sep=',')

##############################################################################
# Time to Process Analysis
print('Time to Process Script %.4f Seconds') % (time.time() - t0)
