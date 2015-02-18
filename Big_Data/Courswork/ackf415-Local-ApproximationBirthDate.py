##########################################################
# INM 432 - Big Data - Coursework 1
# (C) 2014 Daniel Dixey
# 26/11/2014
##########################################################

##########################################################
################# Import Libraries #######################
# Import various elememts of the Pyspark Module
from pyspark import SparkContext 
from pyspark.conf import SparkConf
from pyspark.mllib.regression import LabeledPoint, LassoWithSGD
from pyspark.mllib.linalg import SparseVector
# Import OS for Traversing Directories
import os
# Import Numpy for Mathematical Functions
import numpy as np
# Import Time for Measuring Processing Times
from time import time
# Import Add for use with the Reduce() functions
from operator import add
# Import Regular Expression
import re
# Import String Module
import string
# Import an XML Parser
import xml.etree.ElementTree as ET
# Import ast Helpers Module
import ast
# Import the Random Module
import random
# Import the Defaultdic sub module
from collections import defaultdict
# Import Pandas for Analysis
import pandas as pd
##########################################################

################ Coursework Question 1A ##################
########### START - Traversing Directories ###############
# Extract the File Name - to use as the Folder name when saving Pickles
def FileExtract(x):
    File =  x[x.rfind('/')+1:]   
    return File[:-4]
# Traverse Directories Recursively to create a list of files
def getFileList(directory):
    fileList = []
    fileSize = 0
    folderCount = 0
    # For Loop to Cycle Through Directories 
    for root, dirs, files in os.walk(directory):
        folderCount += len(dirs)
        for file in files:
            f = os.path.join(root,file)    
            # Filter out Files greater than 1 mb and duplicate books
            if (os.path.getsize(f) < 1048576) & ('-' not in FileExtract(f)) & (f != directory):
                fileSize = fileSize + os.path.getsize(f)
                fileList.append(f)
    # Print to Check Data has been located correctly
    print('################################################')
    print('############ Traversing Directories ############\n')
    print('Directory = %s') % (directory)
    print("Total Size is {0} bytes".format(fileSize))
    print('Number of Files = %i') % (len(fileList))
    print('Total Number of Folders = %i') % (folderCount)
    print('################################################\n')
    return fileList
# Traverse Directories Recursively to create a list of files
def getFileListv2(directory):
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
    print('Directory = %s') % (directory)
    print("Total Size is {0} bytes".format(fileSize))
    print('Number of Files = %i') % (len(fileList))
    print('Total Number of Folders = %i') % (folderCount)
    print('################################################\n')
    return fileList
# Find the list of Directories not files - need for loading Pickles
def getDirectory(x):
    directories = []
    for dirPath, dirNames, fileNames in os.walk(x, topdown=False):
        if len(dirPath) > len(x):
            directories.append(dirPath)
    return directories
############# END- Traversing Directories ################
##########################################################


################ Coursework Question 1B ##################
# Finding the Header Funtion
def findHeader(x):
    headStart = ['*** start of this project gutenberg',
    '***start of the project gutenberg',
    '**welcome to the world of free plain vanilla electronic texts**']
    # For Loop to Iterate
    if headStart[0] in x:
        return True
    elif headStart[1] in x:
        return True
    elif headStart[2] in x:
        return True
    else:
        return False
# Finding the Footer Funtion
def findFooter(x):
    footerStart = ['*** end of the project gutenberg',
    '*** end of this project gutenberg',
    '***end of the project gutenberg']
    if footerStart[0] in x:
        return True
    elif footerStart[1] in x:
        return True
    elif footerStart[2] in x:
        return True
    else:
        False
##########################################################
    

################ Coursework Question 1C ##################
########## START - Extracting Ebook Numbers ##############
# Extract the Ebook Number
def findEBookNo(x):
    # If Statement to Identify the 3 Classification I have found in the Main Dataset
    if 'ebook' in x:
        return ebookNo(x)
    elif 'etext' in x:
        return etextNo(x)
    elif 'eook' in x:
        return eookNo(x)     
# Ebook Variations   
def ebookNo(x): 
    x = re.search(re.escape('ebook')+"(.*)"+re.escape(']'),x).group(1)
    # Remove Hash from Value
    re1='.*?'; re2='(\\d+)'
    rg = re.compile(re1+re2,re.IGNORECASE|re.DOTALL)
    m = rg.search(x)
    if m:
        return m.group(1)
# EText Variations
def etextNo(x):
    x = re.search(re.escape('etext')+"(.*)",x).group(1)
    # Remove Hash Value
    re1='.*?'; re2='(\\d+)'
    rg = re.compile(re1+re2,re.IGNORECASE|re.DOTALL)
    m = rg.search(x)
    if m:
        return m.group(1)
# ebook Variation
def eookNo(x):
    re1='.*?'; re2='\\d+'; re3='.*?'; re4='\\d+'; re5='.*?'; re6='(\\d+)'
    rg = re.compile(re1+re2+re3+re4+re5+re6,re.IGNORECASE|re.DOTALL)
    m = rg.search(x)
    if m:
        return m.group(1)
########## END - Extracting Ebook Numbers ##############
########################################################


################ Coursework Question 1D ################
########## START - Removing Header & Footer ############
def processFile(x, i, fileEbook, stop_words):
    # Start Process Watch
    start = time()             
    # Import File and convert to Lower Text
    text = sc.textFile(x) \
            .zipWithIndex() \
            .map(lambda (x,y): (x.lower(),y))
    # Find Number of Lines in Text
    text_len = text.count()       
    # Find the Last Line of Header          
    headerLine = text.filter(lambda (x,y): findHeader(x)==True) \
                    .map(lambda (x,y): y) \
                    .collect()   
    # Check if Header is Found
    if len(headerLine)>=0:
        headerLine=headerLine[0]
    else:
        headerLine=0
    # Find First line of the Footer
    footerLine = text.filter(lambda (x,y): (findFooter(x))==True) \
                    .map(lambda (x,y): y) \
                    .collect()    
    # Check if Footer is found
    if len(footerLine)>0:
        footerLine=footerLine[0]
    else:
        footerLine=text_len            
    # Find the Ebook Number
    Ebook = text.filter(lambda (x,y): ('[ebook' in x) | ('[etext' in x) | ('[eook' in x)) \
                .map(lambda (x,y): findEBookNo(x)) \
                .collect()            
    # Extract the File name from Root
    filename = FileExtract(x)            
    # Subset of Text
    text =  text.filter(lambda (x,y): y > headerLine) \
                .filter(lambda (x,y): y < footerLine)         
    # Extract the list of Word Frequency pairs per file (as an RDD)
    text =  text.flatMap(lambda (x,y): (re.split('\W+',x))) \
                .map(lambda x: (str(x),1)) \
                .filter(lambda x: x[0] not in stop_words) \
                .filter(lambda x: (len(x[0]) < 15)) \
                .filter(lambda x: (len(x[0]) > 1)) \
                .reduceByKey(add)
    # Maximum Term Frequency by File           
    MaxFreq =   text.map(lambda (x): (filename,x[1])) \
                    .reduceByKey(max) \
                    .map(lambda (x,y): y) \
                    .collect()
    # Check if All the Elements have been found
    if (len(MaxFreq)==0) | (len(filename)==0):              
        return None
    else:
        MaxFreq = MaxFreq[0]
    # Find the Term Frequency Files
    text.map(lambda (x): (filename, [x[0],x[1],float(MaxFreq)])) \
                    .map(lambda (x,y): (x, [y[0], float(y[1]/y[2])])) \
                    .map(lambda (x,y): (x,[[y[0],y[1]]])) \
                    .repartition(8) \
                    .reduceByKey(add) \
                    .saveAsPickleFile('/home/dan/Desktop/IMN432-CW01/Word_Freq/' + str(filename))
    # Stop Clock
    fin = time()                        
    # Log File Details - For Analysis
    fileEbook += [[filename, Ebook[0], headerLine, footerLine, fin - start]]
    # Return Text for Processing
    return 'Complete'
########## END - Removing Header & Footer ##############
########################################################

################## Question 1F #########################
######## START - Calculate the TF.IDF values ###########
# Creating a Hashed Vector
def hashVector(fileName, tf_idf, vsize):
    vec = [0] * vsize # initialise vector of vocabulary size
    for wc in tf_idf:
        i = hash(wc[0]) % vsize # get word index
        vec[i] = vec[i] + wc[1] # add count to index
    return (fileName, vec)
######### END - Calculate the TF.IDF values ############
########################################################


################## Question 3B #########################
########### START - Create Subsets of Data  ############
# Obtain a list of EbookID and Birth Date of Author
def xmlExtractDate(files, table):
    if '.rdf' in files:
        # Import into the XML Parser
        root = ET.parse(files).getroot()
        # Getting the Book ID        
        ebookChild = root.find('{http://www.gutenberg.org/2009/pgterms/}ebook')    
        book = ebookChild.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        rgID = re.compile('.*?(\\d+)', re.IGNORECASE|re.DOTALL)
        ebookID = rgID.search(book).group(1)
        try:
            # Obtaining the Birth Date
            creatorDetails = ebookChild.find('{http://purl.org/dc/terms/}creator')
            creatorDetailsAgent = creatorDetails.find('{http://www.gutenberg.org/2009/pgterms/}agent')
            Date = creatorDetailsAgent.find('{http://www.gutenberg.org/2009/pgterms/}deathdate')
            table.append([ebookID, Date.text])
        except AttributeError:
            x = 1
    return table
########### END - Find Most Popular Subjects ############
#########################################################


################## Other Functions #####################
# Map the Dates onto the RDD
def mappingDates(x, table):
    # Loop through array
    for values in table:
        # If Ebook Number found
        if x in values:
            return values[1]
        # If not Found
    return 'Not Found'
########################################################

########################################################
################### Main Spark #########################
if __name__ == "__main__":
    # Time of the Process
    start_time_overall = time()   
    # Print Header to Output File
    print('################################################')
    print('############ Coursework1 - ackf415 #############')
    print('############# IMN430 - 26/11/2014 ##############')
    print('################################################\n')
    # Get Hierarchy of Data Structure
    directory = ['/home/dan/Desktop/IMN432-CW01/text-part/'         \
                ,'/home/dan/Desktop/IMN432-CW01/TF_IDF/'            \
                ,'/home/dan/Desktop/IMN432-CW01/Word_Freq/'         \
                ,'/home/dan/Desktop/IMN432-CW01/IDF/'               \
                ,'/home/dan/Desktop/IMN432-CW01/IDF/IDF-Pairs'      \
                ,'/home/dan/Desktop/IMN432-CW01/IDF'                \
                ,'/home/dan/Desktop/IMN432-CW01/TF_IDF/TF_IDF_File' \
                ,'/home/dan/Desktop/IMN432-CW01/processXML/'        \
                ,'/home/dan/Desktop/IMN432-CW01/meta/'              \
                ,'/home/dan/Desktop/IMN432-CW01/TF_IDF'             \
                ,'/home/dan/Desktop/IMN432-CW01/processXML/Subject' \
                ,'/home/dan/Spark_Files/Books/stopwords_en.txt']
    allFiles = getFileList(directory[0])
    # Find the Number of Files in the Directory
    numFiles = len(allFiles)
    # Create Spark Job Name and Configuration Settings
    config = SparkConf().setMaster("local[*]")
    config.set("spark.executor.memory","5g")
    sc = SparkContext(conf=config, appName="ACKF415-Coursework-1")
    # Create a File Details List
    N = numFiles; fileEbook = []
    print('################################################')
    print('###### Process Files > Word Freq to Pickle #####\n')
    # Start Timer
    WordFreq_Time = time()    
    # Pickled Word Frequencies
    pickleWordF = getFileList(directory[2])
    # Ascertain if Section has already been completed
    if len(pickleWordF) < 1:
        print 'Creating Work Freq Pickles and RDDs \n'
        # Import the Stop and Save as a List
        stop_words = sc.textFile(directory[11]).flatMap(lambda x: re.split(',',x)).collect()
        # Loop Through Each of the Files to Extract the Required Parts    
        for i in np.arange(0, N): # Change to specify the number of Files to Process      
            # Process the Files and return the Individual Document Frequency
            text = processFile(allFiles[i], i, fileEbook, stop_words)
    else:
        print 'Using Pre-Pickled Files\n'
    # End Timer for this phase
    WordFreq_Time = time() - WordFreq_Time
    print('############ Processing Completed ##############')
    print('################################################\n')

    print('################################################')
    print('############## Word Freq to IDF RDD ############\n')
    # Start Timer
    IDF_Time = time()
    # Ascertain if Section has already been completed
    if len(getDirectory(directory[3])) < 1:
        allFolders = getDirectory(directory[2])
        # Load in Word Frequency Pickles into one RDD
        IDF = sc.union([sc.pickleFile(i) for i in allFolders])
        # Rearrange RDD into correct the correct format
        IDF = IDF.flatMap(lambda (x,y): [(pair[0], [[x, str(pair[1])]]) for pair in y]) \
                 .reduceByKey(add) \
                 .map(lambda (x,y): (x,len(y),float(N),y)) \
                 .map(lambda (x,y,z,a): (x,np.log2(z/y),a)) \
                 .repartition(8)
        # Save IDF RDD as a Pickle File 
        IDF.saveAsPickleFile(directory[4],50)
    else:
        print 'Using Pre-Pickled Files\n'
    # End Timer for this phase  
    IDF_Time = time() - IDF_Time
    print('############ Processing Completed ##############')
    print('################################################\n')
    
    print('################################################')
    print('################ IDF to TF.IDF #################\n')    
    # Start Timer    
    TFIDF_Time = time()
    # Ascertain if Section has already been completed
    if len(getDirectory(directory[1])) < 1:
        allFolders = getDirectory(directory[5])
        # Load in IDF Pickles into one RDD
        IDF = sc.union([sc.pickleFile(i) for i in allFolders])
        # Rearrange RDD into correct the correct format
        TF_IDF = IDF.map(lambda (x,y,z): (x,[[pair[0],y*ast.literal_eval(pair[1])] for pair in z])) \
                    .flatMap(lambda (x,y): [(pairs[0],[[x,str(pairs[1])]]) for pairs in y]) \
                    .reduceByKey(add) \
                    .map(lambda (x,y): (x,[[pairs[0], ast.literal_eval(pairs[1])] for pairs in y])) \
                    .repartition(8)
        # Save TF.IDF RDD as a Pickle File
        TF_IDF.saveAsPickleFile(directory[6], 50)
    else:
        print 'Using Pre-Pickled Files\n'
    # End Timer for this phase
    TFIDF_Time = time() - TFIDF_Time
    print('############ Processing Completed ##############')
    print('################################################\n')  
    
    print('################################################')
    print('############## Process XML Files ###############\n')
    # Start Timer    
    XML_Time = time()
    # Location of Meta Data
    file_loc = '/home/dan/Desktop/IMN432-CW01/meta/'
    files = getFileList(file_loc)
    # Loop through Files
    table = []
    for location in files:
        table = xmlExtractDate(location, table)
    # End Timer for this phase
    XML_Time = time() - XML_Time
    print('\n############ Processing Completed ##############')
    print('################################################\n')    
    ######################################################
    
    print('################################################')
    print('######### Testing and Building Models ##########\n')    
    # Start Stopwatch    
    modelTime = time()
    # Read in TF.IDF File
    if len(getDirectory(directory[9])) == 1:
        # Get a list of the directory where the TF.IDF Vector is stored as a Pickle
        tf_idf_directory = getDirectory(directory[9])
        # Read in all the Files
        tf_idf = sc.union([sc.pickleFile(i) for i in tf_idf_directory])
        # Repartition the TF_IDF RDD for speed
        tf_idf = tf_idf.repartition(8)
        # Ebooks in Use in Vector
        tf_idf_ebooks = tf_idf.keys().collect()
        # Import Vector of Ebook and Ebook Ids into Spark
        authorDate = sc.parallelize(table)
        # Get the Ebook IDs filter for only books in Text- Part
        authorDate = authorDate.filter(lambda x: x[0] in tf_idf_ebooks)
        authorDate1 = authorDate.keys().collect() 
        authorDate2 = authorDate.values().collect() 
        # Check the Other way to exclude books in Text Part where the Date was not found
        tf_idf = tf_idf.filter(lambda x: x[0] in authorDate1)
        # Make a list of Birthdates to Map into the tf_idf vector
        authorDate = authorDate.collect()
        # Create a Test and Training Sets
        TestRDD = tf_idf.sample(False, 0.2)
        TrainingRDD = tf_idf.subtractByKey(TestRDD)
        # Transform the Data
        TestRDD = TestRDD.map(lambda x: (mappingDates(x[0], authorDate), x[1]))
        TrainingRDD = TrainingRDD.map(lambda x: (mappingDates(x[0], authorDate), x[1]))
        # Create Hashed Vectors
        TestRDD = TestRDD.map(lambda x: (hashVector(x[0], x[1], 10000)))
        TrainingRDD = TrainingRDD.map(lambda x: (hashVector(x[0], x[1], 10000)))
        # Create Labelled Points of Each of the Vectors
        TrainingRDD = TrainingRDD.map(lambda (f,x): LabeledPoint(f,x))
        # Train Model on The Training Set        
        model = LassoWithSGD.train(TrainingRDD)
        # Test the Model on the Test Set  
        predictions = []
        TestRDD_Array = TestRDD.values().collect()
        for i in np.arange(0, len(TestRDD_Array)):
            Prediction_Label = model.predict(np.array(TestRDD_Array[i]))
            predictions.append(Prediction_Label)
        
        TestRDD_Array_Label = TestRDD.keys().collect()
        for i in np.arange(0,len(TestRDD_Array_Label)):
            print TestRDD_Array_Label[i], predictions[i]

    # Stop Watch
    modelTime = time() - modelTime
    print('\n############ Processing Completed ##############')
    print('################################################\n')    
    ######################################################
    
    print('################################################')
    print('##### Display Overall Time and Statistics ######')
    print('################################################')
    print('Time to Process TF Pickles:     %.3f Seconds') % (WordFreq_Time)
    if len(fileEbook) > 0:
        print('Average Time to Process Files:  %.3f Seconds') % (np.mean(np.array([i[4] for i in fileEbook])))
    print('Time to Process IDF Pickle:     %.3f Seconds') % (IDF_Time)
    print('Time to Process TF.IDF Pickle:  %.3f Seconds') % (TFIDF_Time)
    print('Time to Process XML Files:      %.3f Seconds' % (XML_Time))
    print('Time to Test and Build Model:  %.3f Seconds' % (modelTime))
    print('Total Run Time:                 %.3f Seconds') % (time() - start_time_overall)
    print('################################################')
    ######################################################

    # Disconnect from Spark
    sc.stop()
########################################################
