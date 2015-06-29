# Import Statements
from pymongo import MongoClient
import os
import sys
import pymongo
from bson import BSON
from bson import json_util
import cPickle as pickle

# Connect to Mongo
def createCON():
    try:
        client = MongoClient('10.0.9.164', 27017) # Connect to the Server
        db = client['NEWS'] # Connect to the News Database
    except:
        print('Error: Unable to Connect')
        connection = None
    return db.News

# Query the database and convert BSON to Dictionary 
def cursorDict( cursor ):
    Data = cursor.find({'objectClass': 'NEWS'}) # Query Database
    dictionary = {} # Create an empty Dictionary
    for each_doc in Data:
        tempList = []
        for each_attribute in each_doc:
            if each_attribute == '_id':
                _docID = each_doc[each_attribute]
            tempList.append([each_attribute, each_doc[each_attribute]] )
        dictionary[_docID] = dict( tempList )
    return dictionary

# Save the Dictionary to Pickle for local use
def convertPickle( dictionary ):
    with open(os.getcwd() + '/JSON_File.p', 'wb') as fp:
        pickle.dump(dictionary, fp)

# Save local version of collection
def collectData():
    connectionNEWS = connection()   # Connect the Database
    Data = cursorDict( connectionNEWS )   # Convert Mongo Cursor to Python Dictionary
    convertPickle( Data ) # Save to pickle for later use

# Load Dictionary file from Pickle        
def loadLocal():
    with open(os.getcwd() + '/JSON_File.p', 'rb') as fp:
        data = pickle.load(fp)
    return data

# Tokenize, punctuation, lowercase
# TF/IDF
# Import Lists for CSV
# Annotate, ngrams
# Filter Datasets

if __name__ == "__main__":
    Data = loadLocal()
    print len(Data)
