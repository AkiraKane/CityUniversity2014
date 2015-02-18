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
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
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
    if len(headerLine)>0:
        headerLine = headerLine[0]
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
                    .saveAsPickleFile('/data/student/ackf415/Word_Freq/' + str(filename))
    # Stop Clock
    fin = time()                        
    # Log File Details - For Analysis
    fileEbook += [[filename, Ebook, headerLine, footerLine, fin - start]]
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

################## Question 2 ##########################
########### START - Processing XML Files ###############
def xmlExtract(x, table):
    # Process only RDF Files
    if '.rdf' in x:
        root = ET.parse(x).getroot()
        # Get the Ebook Child off the Root
        ebookChild = root.find('{http://www.gutenberg.org/2009/pgterms/}ebook')
        # Obtain the Namespace
        book = ebookChild.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        # Compile Regular Expression to Extract ID Number
        rgID = re.compile('.*?(\\d+)', re.IGNORECASE|re.DOTALL)
        # Extract the Ebook ID
        ebookID = rgID.search(book).group(1)
        # Find all the Child off the Ebook object to get all the Subjects
        allSubjects = ebookChild.findall('{http://purl.org/dc/terms/}subject')
        # Loop through the Subjects and save into a list
        for subject in allSubjects:
            # Find the Values
            x = subject.find('*').find('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}value')
            # Add to a list
            table.append([ebookID, x.text])
    # Return the updated List
    return table
############ END - Processing XML Files ################
########################################################

################## Question 3A #########################
######### START - Find Most Popular Subjects ###########
# Find Top Subjects and corresponding files
def processXML(table):
    # Get a list of the directory where the TF.IDF Vector is stored as a Pickle
    tf_idf_directory = getDirectory('/data/student/ackf415/TF_IDF' )
    # Read in all the Files
    ebookNumbers = sc.union([sc.pickleFile(i) for i in tf_idf_directory])
    # Get Ebook numbers for analysis and also remove Null Values
    ebookNumbers = ebookNumbers.map(lambda (x,y): x) \
                             .collect()
    # Find the Top 10 Subjects
    top_10_subjects = sc.parallelize(table) \
                .filter(lambda x: x[0] in ebookNumbers) \
                .map(lambda x: (x[1],1)) \
                .reduceByKey(add)
    # Get a List of all the of the Top 10 Subjects
    Subjects_Top_10 = top_10_subjects.takeOrdered(10, key=lambda x: -x[1])
    # For Loop for Print the Values to the Output
    for i in Subjects_Top_10:
        # Print Subject
        print i[0]
    # Create DataFrame for Saving the Data to a CSV for Review later
    df = pd.DataFrame(Subjects_Top_10, columns=['Subject', 'Number of Files with Subject'])
    df.to_csv('IMN432-CW1/Top_10_Subjects.csv', sep=',')                
    # Save Locatio of top 10 Subjects
    savePath = '/data/student/ackf415/processXML/'
    # Loop through to Find the Files where the Subjects are contained
    file_list = sc.parallelize(table)  
    for i in np.arange(0,10):
        # Save the Subject Lable to a Variable for use in the filter
        Label = Subjects_Top_10[i]        
        # Save the Files in Each Subject to a Pickle File
        file_list.filter(lambda x: x[0] in ebookNumbers) \
                                        .filter(lambda (x): x[1] == Label[0]) \
                                        .map(lambda x: (x[1], [x[0]])) \
                                        .reduceByKey(add) \
                                        .saveAsPickleFile(savePath + 'Subject' + str(i))
    # Return Two Objects for Testing ONLY
    return Subjects_Top_10
########## END - Find Most Popular Subjects ############
########################################################

################## Question 3B #########################
########### START - Create Subsets of Data  ############
def makeSets(RDD, trainingList, testList, subject_lists, hashsize):
    # Create the training Set - Correctly at the data as labelled Points
    trainingRDDHashed = RDD.filter(lambda (x,y): x in trainingList) \
                            .map(lambda (x,y): (checkFile(x, subject_lists), y)) \
                            .map(lambda (f, wl): (hashVector(f,wl,hashsize)))
    trainingRDD = trainingRDDHashed.map(lambda (f,x): LabeledPoint(f,x))
    # Create the validation Set - Correctly at the data as labelled Points
    testRDDHashed = RDD.filter(lambda (x,y): x in trainingList) \
                            .map(lambda (x,y): (checkFile(x, subject_lists), y)) \
                            .map(lambda (f, wl): (hashVector(f,wl,hashsize)))
    testRDD = testRDDHashed.map(lambda (f,x): LabeledPoint(f,x))
    # Retun the data into the Main function
    return trainingRDD, testRDD, trainingRDDHashed, testRDDHashed
# Check if Files is in Subject  
def checkFile(x, listFiles):
    # Check if Files is in List
    if x in listFiles[0]:
        # Return 1 if in Subject List
        return 1
    else:
        # Return 0 if not
        return 0
def getSetList(x):
    # Take 80% of the possible values for analysis
    trainingList = random.sample(x, int(len(x)*0.7))
    # Extract the Difference between the Full List and the Training List
    testList = list(set(x).difference(trainingList))
    # Sub-devide the list into two list
    return trainingList, testList
# Naive Bayes Models and Results
def naiveBayes(trainingRDD, trainingRDDHashed, testRDDHashed):
    # Naive Bayes
    trainedModel = NaiveBayes.train(trainingRDD, 1.0)
    # Test on Validation and Test Sets
    resultsValidation = trainingRDDHashed.map(lambda (l,v):  ((l,trainedModel.predict(v)),1)).reduceByKey(add).collectAsMap()
    resultsTest = testRDDHashed.map(lambda (l,v):  ((l,trainedModel.predict(v)),1)).reduceByKey(add).collectAsMap()
    # Get Counts
    nFilesV = trainingRDDHashed.count(); nFilesT = testRDDHashed.count()
    # Create a dictionary of the Values
    resultsValidation = defaultdict(lambda :0, resultsValidation)
    resultsTest = defaultdict(lambda :0, resultsTest)
    # Get F-Score and Accuracy Values
    AccuracyV, fScoreV = getAccuracy(resultsValidation, nFilesV)
    AccuracyT, fScoreT = getAccuracy(resultsTest, nFilesT)
    # Print Results
    print('   Results for Naive Bayes')
    print('      Training Set: %.3f and F-Score: %.3f') % (AccuracyV, fScoreV)
    print('      Test Set: %.3f and F-Score: %.3f') % (AccuracyT, fScoreT)
    # Return the Result List
    return AccuracyV, fScoreV, AccuracyT, fScoreT
## Decision Tree Models and Results
def decisionTree(trainingRDD, trainingRDDHashed, testRDDHashed, testRDD):
    # Get size of RDD
    nFilesV = trainingRDDHashed.count(); nFilesT = testRDDHashed.count()    
    # Train the Decision Tree Model
    trainedModel = DecisionTree.trainClassifier(trainingRDD, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=2, maxBins=3)
    # Test the Model on the Training Set    
    predictions = trainedModel.predict(trainingRDD.map(lambda x: x.features))
    labelsAndPredictions = trainingRDD.map(lambda lp: lp.label).zip(predictions).countByValue()
    # Map to Dictionary for obtaining Results
    resultsValidation = defaultdict(lambda :0, labelsAndPredictions)
    nFilesV = trainingRDDHashed.count(); nFilesT = testRDDHashed.count()
    # Get F-Score and Accuracy Value
    AccuracyV, fScoreV = getAccuracy(resultsValidation, nFilesV)
    # Test the Model on the Test Set  
    predictions = trainedModel.predict(testRDD.map(lambda x: x.features))
    labelsAndPredictions = testRDD.map(lambda lp: lp.label).zip(predictions).countByValue()
    # Map to Dictionary for obtaining Results
    resultsTest = defaultdict(lambda :0, labelsAndPredictions)
    AccuracyT, fScoreT = getAccuracy(resultsTest, nFilesT)
    # Print Results
    print('   Results for Decision Tree')
    print('      Training Set: %.3f and F-Score: %.3f') % (AccuracyV, fScoreV)
    print('      Test Set: %.3f and F-Score: %.3f') % (AccuracyT, fScoreT)
    # Return the Result List
    return AccuracyV, fScoreV, AccuracyT, fScoreT
# Logistics Regression Models and Resuts
def logisticRegression(trainingRDD, trainingRDDHashed, testRDDHashed):
    # Train a Naive Bayes Model
    trainedModel = LogisticRegressionWithSGD.train(trainingRDD, iterations=100, miniBatchFraction=1.0, regType="l2", intercept=True, regParam=1.0)
    # Test on Validation and Test Sets
    resultsValidation = trainingRDDHashed.map(lambda (l,v):  ((l,trainedModel.predict(v)),1)).map(lambda (x,y): (checkState(x),y)).reduceByKey(add).collectAsMap()
    resultsTest = testRDDHashed.map(lambda (l,v):  ((l,trainedModel.predict(v)),1)).map(lambda (x,y): (checkState(x),y)).reduceByKey(add).collectAsMap()
    # Get Counts
    nFilesV = trainingRDDHashed.count(); nFilesT = testRDDHashed.count()
    # Create a dictionary of the Values
    resultsValidation = defaultdict(lambda :0, resultsValidation)
    resultsTest = defaultdict(lambda :0, resultsTest)
    # Get F-Score and Accuracy Values
    AccuracyV, fScoreV = getAccuracy(resultsValidation, nFilesV)
    AccuracyT, fScoreT = getAccuracy(resultsTest, nFilesT)
    # Print Results
    print('   Results for Logistic Regression')
    print('      Training Set: %.3f and F-Score: %.3f') % (AccuracyV, fScoreV)
    print('      Test Set: %.3f and F-Score: %.3f') % (AccuracyT, fScoreT)
    # Return the Result List
    return AccuracyV, fScoreV, AccuracyT, fScoreT
# Check if Prediction is greater than 0
def checkState(x):
    if x[1] > 0:
        return (x[0],1)
    else:
        return (x[0],0)
########### END - Find Most Popular Subjects ############
#########################################################


################## Other Functions #####################
# Funcion to Remove punctuation from Text - Not used but if Required
def punctuationWord(x):
    # Remove Punctuation from Word
    for letter in string.punctuation:
        x = x.replace(letter,'')
    return str(x)
# print the different performance metrics
def getAccuracy(resDict, count):
    # Calculate quadrants in Confusion Matrixs
    truePos = resDict[(1,1,)]; falsePos = resDict[(0,1,)]
    trueNeg = resDict[(0,0,)]; falseNeg = resDict[(1,0,)]
    # Calculate the Accuracy
    Accuracy = (float(truePos+trueNeg)/count)*100
    # Dealing with Zero and NoneType Values
    if (truePos is None) | (falsePos is None) | (trueNeg is None) | (falseNeg is None) | \
                (truePos == 0) | (falsePos == 0) | (trueNeg == 0) | (falseNeg == 0):
        fScore = -1
    else:
        # Calculate Precision and Reclass to Find the F-Score
        Recall = float(truePos)/(truePos+falseNeg) # Recall
        Precision = float(truePos)/(truePos+falsePos) # Precision
        # Calculate F-Score as Mentioned in the PDF
        fScore = (2*Precision*Recall)/(Precision + Recall) 
    # Return Accuracy for Print in the Output and the Result Listing
    return Accuracy, fScore
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
    directory = ['/data/extra/gutenberg/text-part/'         \
                ,'/data/student/ackf415/TF_IDF/'            \
                ,'/data/student/ackf415/Word_Freq/'         \
                ,'/data/student/ackf415/IDF/'               \
                ,'/data/student/ackf415/IDF/IDF-Pairs'      \
                ,'/data/student/ackf415/IDF'                \
                ,'/data/student/ackf415/TF_IDF/TF_IDF_File' \
                ,'/data/student/ackf415/processXML/'        \
                ,'/data/extra/gutenberg/meta/'              \
                ,'/data/student/ackf415/TF_IDF'             \
                ,'/data/student/ackf415/processXML/Subject' \
                ,'IMN432-CW1/stopwords_en.txt']
    allFiles = getFileList(directory[0])
    # Find the Number of Files in the Directory
    numFiles = len(allFiles)
    # Create Spark Job Name and Configuration Settings
    config = SparkConf().setMaster("local[2]")
    config.set("spark.executor.memory","12g")
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
    # Check Pickles have been created
    pickleXML = getFileListv2(directory[7])
    # Ascertain if Section has already been completed
    if len(pickleXML) < 1:
        print 'Creating XML Pickles and RDDs \n'
        file_loc = directory[8]
        files = getFileList(file_loc)
        # Loop through Files
        table = []
        for location in files:
            table = xmlExtract(location, table)
        # Process XML Files
        top_10 = processXML(table)
    else:
        print 'Using Pre-Pickled Files'
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
        # Get Ebook numbers for analysis and also remove Null Values
        ebookNumbers = tf_idf.map(lambda (x,y): x) \
                             .filter(lambda (x): len(x) > 1) \
                             .collect()
        # Results List - [Hashsize, Subject, CrossVal, Model, Training Acc, Training F1, Test Acc, Test F1, Time]
        resultList = []
        # Define the Hashsize
        hashsizeVec = [1000]
        for hashsize in hashsizeVec:
            print('Hashsize in Use: %i') % (hashsize)
            # Loop Through Subjects to build models
            for i in np.arange(0,10):
                print('\nDealing with Subject #%i') % (i+1)
                subject_lists = sc.pickleFile(directory[10] + str(i) + '/') \
                                  .map(lambda x: x[1]) \
                                  .collect()
                # Cross Validation Step
                for k in np.arange(1,11):
                    print('\n Cross Validation: %i') % (k)
                    # Copy the Hashed RDD
                    modelSet = tf_idf
                    # Create List for filtering the Data
                    trainingList, testList = getSetList(ebookNumbers)
                    # Create Sets of data into the correct format...
                    trainingRDD, testRDD, trainingRDDHashed, testRDDHashed = makeSets(modelSet, trainingList, testList, subject_lists, hashsize)
                    # Naive Bayes ML Algorithm
                    timeIt = time()
                    AccuracyV, fScoreV, AccuracyT, fScoreT = naiveBayes(trainingRDD, trainingRDDHashed, testRDDHashed)
                    resultList.append([hashsize, i+1, k, 1, AccuracyV, fScoreV, AccuracyT, fScoreT, time() - timeIt])
                    # Decision Tree ML Algorithm
                    timeIt = time()
                    AccuracyV, fScoreV, AccuracyT, fScoreT = decisionTree(trainingRDD, trainingRDDHashed, testRDDHashed, testRDD)
                    resultList.append([hashsize, i+1, k, 2, AccuracyV, fScoreV, AccuracyT, fScoreT, time() - timeIt])
                    # Logistic Regression ML Algorithm
                    timeIt = time()
                    AccuracyV, fScoreV, AccuracyT, fScoreT = logisticRegression(trainingRDD, trainingRDDHashed, testRDDHashed)
                    resultList.append([hashsize, i+1, k, 3, AccuracyV, fScoreV, AccuracyT, fScoreT, time() - timeIt])
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
    print('Time to Test and Build Models:  %.3f Seconds' % (modelTime))
    print('Total Run Time:                 %.3f Seconds') % (time() - start_time_overall)
    print('################################################')
    ######################################################
    # Save Results to a Pandas Daframe and then to a file
    df = pd.DataFrame(resultList, columns=['Hashsize', 'Subject', 'Cross Validation Fold', 'Model', 'Training Accuracy', 'Training F-Score', 'Test Accuracy', 'Test F-Score', 'Time to Evaluate Model'])
    df.to_csv('IMN432-CW1/Results' + str(hashsize) + '.csv', sep=',')  
    # Disconnect from Spark
    sc.stop()
########################################################
