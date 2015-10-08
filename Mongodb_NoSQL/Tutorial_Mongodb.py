# Mongodb Tutorial
# Daniel Dixey
# 6/5/2015

# Tutorial Script

# Import Modules
from pymongo import MongoClient
import pymongo

# Define the Client Connection
client = MongoClient(host='localhost',
                     port=27017,
                     connect=True)
# Print Client Settings
print('\n%s\n') % (client)

# Assign a database to an Object
db = client.test

# Print the Names of the Collections
for names in db.collection_names():
    print('Collection Name: %s') % (names)

# Assign a collection to an Object
Test_Data = db['Test_Data']

# Print Number of Documents
print('\nNumber of Documents: %d') % (Test_Data.find().count())

# Create a Cursor value
cursor = db.restaurants.find()


def printing(cursor):
    print('\n')
    # Print each document
    for document in cursor:
        print(document)

# LOGICAL COMMANDS

# Logical AND
Logical_AND = Test_Data.find({"cuisine": "Italian",
                              "address.zipcode": "10075"})

# Printing
# printing(Logical_AND)

# Logical OR
Logical_OR = Test_Data.find({"$or": [{"cuisine": "Italian"},
                                     {"address.zipcode": "10075"}]
                             })

# Printing
# printing(Logical_OR)

# Less Than Operator ($lt)
Less_Than = Test_Data.find({"grades.score": {"$lt": 10}})

# Printing
# printing(Less_Than)

# Greater Than Operator ($gt)
Greater_Than = Test_Data.find({"grades.score": {"$gt": 30}})

# Printing
# printing(Greater_Than)

# Sorting
Sorting = Test_Data.find().sort([
    ("borough", pymongo.ASCENDING),
    ("address.zipcode", pymongo.DESCENDING)
])

# Printing
# printing(Sorting)

# AGGREGATION

Aggregation = Test_Data.aggregate([
    {"$group": {"_id": "$borough", "count": {"$sum": 1}}}
])

# Printing
printing(Aggregation)

# Additional Information  - TO DO

# Comparision to SQL
# http://docs.mongodb.org/manual/reference/sql-aggregation-comparison/

# Mongodb
# http://docs.mongodb.org/manual/reference/operator/aggregation/interface/

# Indexes with PyMongo - TO DO
