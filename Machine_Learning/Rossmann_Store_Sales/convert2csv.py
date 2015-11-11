#!/usr/bin/python

# Created: 11/10/2015
# Dan Dixey

import json
import pandas as pd

# Read in the Results file
json_data = json.loads(open('allResults.json').read())

# Extract each algorithms Data
randomForest = json_data[0]
bayesRidge = json_data[1]

# Get the Results and Params
randomForest_values = randomForest['values']
randomForest_parameters = randomForest['params']

# Get the Results and Params
bayesRidge_values = bayesRidge['values']
bayesRidge_parameters = bayesRidge['params']

# Empty Lists to store Data in
randomForest_results = []
bayesRidge_results = []

for i in range(50):
    randomForest_results.append([abs(randomForest_values[i])] + randomForest_parameters[i].values())
    bayesRidge_results.append([abs(bayesRidge_values[i])] + bayesRidge_parameters[i].values())

# Column Names
bayesCol = [u'Error'] + bayesRidge_parameters[0].keys()
rfCol = [u'Error'] + randomForest_parameters[0].keys()

# Save Data to CSV
pd.DataFrame(randomForest_results, columns=rfCol).to_csv('randomForest_results.csv', 
                                                         sep = ',')
pd.DataFrame(bayesRidge_results, columns=bayesCol).to_csv('bayesRidge_results.csv', 
                                                         sep = ',',
                                                         index_label=False)
