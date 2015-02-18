# Import Spark API
from pyspark import SparkContext 
from pyspark.conf import SparkConf
# Import OS for Traversing Directories
import os
# Import Numpy for Mathematical Functions
import numpy as np
# Import Time for Measuring Processing Times
#from time import time
# Import Add for use with the Reduce() functions
from operator import add
# Import Regular Expression
import re
# Import Shutil For Deleting Directories and Files
#import shutil
# Import String Module
#import string
# Import Mindom Module
from xml.dom import minidom
##########################################################

# Function to Obtain a List of Files
def getFileList(directory):
    fileList = []
    fileSize = 0
    folderCount = 0
    # For Loop to Cycle Through Directories 
    for root, dirs, files in os.walk(directory):
        folderCount += len(dirs)
        for file in files:
            f = os.path.join(root,file)
            fileSize = fileSize + os.path.getsize(f)
            fileList.append(f)
    # Print to Check Data has been located correctly
    print('################################################')
    print('############ Traversing Directories ############\n')
    print("Total Size is {0} bytes".format(fileSize))
    print('Number of Files = %i') % (len(fileList))
    print('Total Number of Folders = %i') % (folderCount)
    print('################################################\n')
    return fileList
# Define function to extract XML
def xmlExtract(files, table):
    for direc in np.arange(0,len(files)):
        # Read and Convert XML to String
        file = open(files[direc],'r')
        data = file.read()
        file.close()
        # Ebook Number Retrieval
        ebook = minidom.parseString(data)
        ebook = ebook.getElementsByTagName('pgterms:ebook')[0].toxml()
        ebook = ebook.replace('<pgterms:ebook>','').replace('</pgterms:ebook>','')
        ebook = ebook[:ebook.find('>\n')]
        ebook = (re.split('/',re.split('=',ebook)[1])[1]).replace('"','')
        # Read XML into an Object
        xmldoc = minidom.parse(files[direc])    
        # Navigate onto the Tree to Find the Ebook No.
        pgterms_ebook = xmldoc.getElementsByTagName('pgterms:ebook')[0]
        # Navigate to the Subjects Tree
        subjects = pgterms_ebook.getElementsByTagName('dcterms:subject')
        # Print the Subjects values
        for lines in subjects:
            values = lines.getElementsByTagName('rdf:value')[0].firstChild.data
            table.insert(direc,[ebook,values])
        # Return Table of Ebooks and Subjects
    return table

#######################
if __name__ == "__main__":
    # Location of Files
    file_loc = '/home/dan/Desktop/IMN432-CW01/meta/'
    # Find a Files in given Directory
    files = getFileList(file_loc)
    # Loop through Files
    table = []
    # Extract Data from XML Files
    table = xmlExtract(files, table)
    # Start Spark
    config = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=config, appName="ACKF415-Coursework-1")
    
    data = sc.parallelize(table) \
                .map(lambda x: (x[1],1)) \
                .reduceByKey(add) \
                .sortBy(lambda x: x[1], ascending=False)
    
    for i in data.take(10):
        print i
    # Disconnect
    sc.stop()
