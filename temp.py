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
from pyspark.storagelevel import StorageLevel
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.classification import NaiveBayes
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
# Import Mindom Module
from xml.dom import minidom
# Import ast Helpers Module
import ast
# Import the Random Module
import random
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
            if (os.path.getsize(f) < 1048576) & ('-' not in FileExtract(f)):
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
def processFile(x, i, fileEbook):
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
    filename = FileExtract(allFiles[i])            
    # Subset of Text
    text = text.filter(lambda (x,y): y > headerLine) \
                .filter(lambda (x,y): y < footerLine)         
    # Extract the list of Word Frequency pairs per file (as an RDD)
    text = text.flatMap(lambda (x,y): (re.split('\W+',x))) \
                .map(lambda x: (str(x),1)) \
                .filter(lambda x: (len(x[0]) < 15)) \
                .filter(lambda x: (len(x[0]) > 1)) \
                .reduceByKey(add)
    # Maximum Term Frequency by File           
    MaxFreq = text.map(lambda (x): (filename,x[1])) \
                    .reduceByKey(max) \
                    .map(lambda (x,y): y) \
                    .collect()
    MaxFreq = MaxFreq[0]
    # Find the Term Frequency Files
    text.map(lambda (x): (str(Ebook[0]), [x[0],x[1],float(MaxFreq)])) \
                    .map(lambda (x,y): (x, [y[0], float(y[1]/y[2])])) \
                    .map(lambda (x,y): (x,[[y[0],y[1]]])) \
                    .repartition(8) \
                    .reduceByKey(add) \
                    .saveAsPickleFile('/home/dan/Desktop/IMN432-CW01/Word_Freq/' + str(filename))
    # Save as a Pickle File - THIS IS HOW I WOULD SAVE THE WORD FREQUENCY PAIRS
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
def hashVector(fileName, wordCount, vsize):
    vec = [0] * vsize # initialise vector of vocabulary size
    for wc in wordCount:
        i = hash(wc[0]) % vsize # get word index
        vec[i] = vec[i] + wc[1] # add count to index
    return (fileName, vec)
######### END - Calculate the TF.IDF values ############
########################################################

################## Question 2 ##########################
########### START - Processing XML Files ###############
def xmlExtract(files, table):
    for direc in np.arange(0,len(files)):
        # Read and Convert XML to String
        x = files[direc]
        if x[x.rfind('.')+1:]=='rdf':        
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
                table.append([ebook,values])
    # Return Table of Ebooks and Subjects
    return table
############ END - Processing XML Files ################
########################################################

################## Question 3A #########################
######### START - Find Most Popular Subjects ###########
# Find Top Subjects and corresponding files
def processXML(table):
        # Find the Top 10 Subjects
    top_10_subjects = sc.parallelize(table) \
                .map(lambda x: (x[1],1)) \
                .reduceByKey(add) \
                .sortBy(lambda x: x[1], ascending=False) \
                .map(lambda x: x[0])
    
    # Iterate through to get the Top 10 Subjects
    subjects = {}; num = 0; top_10={}
    # For Loop 
    for i in top_10_subjects.take(10):
        top_10["Subject{0}".format(num)] = i
        num += 1                   
    # Loop through to Find the Files where the Subjects are contained
    num = 0; savePath = '/home/dan/Desktop/IMN432-CW01/Backup/processXML/'
    file_list = sc.parallelize(table)  
    # Save RDD as Files
    for i in top_10:
        subjects[top_10[i]] = file_list.filter(lambda (x): x[1] in top_10[i]) \
                                    .map(lambda x: (x[1], [x[0]])) \
                                    .reduceByKey(add) \
                                    .saveAsPickleFile(savePath + str(i))
        # Increase step by One
        num += 1
    # Return two Objects
    return subjects, top_10
########## END - Find Most Popular Subjects ############
########################################################

################## Question 3B #########################
########### START - Create Subsets of Data  ############
# Check if Files is in Subject  
def checkFile(x, listFiles):
    if x in listFiles:
        return 1
    else:
        return 0
def getSetList(x):
    # Take 80% of the possible values for analysis
    trainingList = random.sample(x, int(len(x)*0.8))
    # Extract the Difference between the Full List and the Training List
    subset = list(set(x).difference(trainingList))
    # Sub-devide the list into two list
    testList, validationList = zip(*[iter(subset)]*int(len(subset)/2)) 
    return trainingList, testList, validationList
########## END - Find Most Popular Subjects ############
########################################################


################## Other Functions #####################
# Funcion to Remove punctuation from Text
def punctuationWord(x):
    # Remove Punctuation from Word
    for letter in string.punctuation:
        x = x.replace(letter,'')
    return str(x)
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
    directory = ['/home/dan/Desktop/IMN432-CW01/text-part/' \
                ,'/home/dan/Desktop/IMN432-CW01/TF_IDF/'    \
                ,'/home/dan/Desktop/IMN432-CW01/Word_Freq/' \
                ,'/home/dan/Desktop/IMN432-CW01/IDF/'       \
                ,'/home/dan/Desktop/IMN432-CW01/IDF/IDF-Pairs' \
                ,'/home/dan/Desktop/IMN432-CW01/IDF'        \
                ,'/home/dan/Desktop/IMN432-CW01/TF_IDF/TF_IDF_File' \
                ,'/home/dan/Desktop/IMN432-CW01/processXML/'\
                ,'/home/dan/Desktop/IMN432-CW01/meta/'      \
                ,'/home/dan/Desktop/IMN432-CW01/TF_IDF'      \
                ,'/home/dan/Desktop/IMN432-CW01/processXML/Subject']
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
        # Loop Through Each of the Files to Extract the Required Parts    
        for i in np.arange(0, N): # Change to specify the number of Files to Process      
            # Process the Files and return the Individual Document Frequency
            text = processFile(allFiles[i], i, fileEbook)
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
        IDF = sc.union([sc.pickleFile(i) for i in allFolders])
        IDF = IDF.flatMap(lambda (x,y): [(pair[0], [[x, str(pair[1])]]) for pair in y]) \
                 .reduceByKey(add) \
                 .map(lambda (x,y): (x,len(y),float(N),y)) \
                 .map(lambda (x,y,z,a): (x,np.log2(z/y),a)) \
                 .repartition(8)
             
        IDF.saveAsPickleFile(directory[4],50)
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
        IDF = sc.union([sc.pickleFile(i) for i in allFolders])
        TF_IDF = IDF.map(lambda (x,y,z): (x,[[pair[0],y*ast.literal_eval(pair[1])] for pair in z])) \
                    .flatMap(lambda (x,y): [(pairs[0],[[x,str(pairs[1])]]) for pairs in y]) \
                    .reduceByKey(add) \
                    .map(lambda (x,y): (x,[[pairs[0], ast.literal_eval(pairs[1])] for pairs in y])) \
                    .repartition(8)
    
        TF_IDF.saveAsPickleFile(directory[6], 50)
    # End Timer for this phase
    TFIDF_Time = time() - TFIDF_Time
    print('############ Processing Completed ##############')
    print('################################################\n')  
    
    print('################################################')
    print('############## Process XML Files ###############\n')
    # Start Timer    
    XML_Time = time()
    # Check Pickles have been created
    pickleXML = getFileList(directory[7])
    # Ascertain if Section has already been completed
    if len(pickleXML) < 1:
        print 'Creating XML Pickles and RDDs \n'
        file_loc = directory[8]
        files = getFileList(file_loc)
        # Loop through Files
        table = []
        # Extract Data from XML Files
        table = xmlExtract(files, table)
        # Process XML Files
        subjects, top_10 = processXML(table)
    else:
        print 'Using Pre-Pickled Files'
    # End Timer for this phase
    XML_Time = time() - XML_Time
    print('\n############ Processing Completed ##############')
    print('################################################\n')    
    ######################################################
    
    print('################################################')
    print('######### Testing and Building Models ##########')    
    # Start Stopwatch    
    modelTime = time()
    # Read in TF.IDF File
    if len(getDirectory(directory[9])) == 1:
        # Get a list of the directory where the TF.IDF Vector is stored as a Pickle
        tf_idf_directory = getDirectory(directory[9])
        # Read in all the Files
        tf_idf = sc.union([sc.pickleFile(i) for i in tf_idf_directory])
        # Get Ebook numbers for analysis and also remove Null Values
        ebookNumbers = tf_idf.map(lambda (x,y): x) \
                             .filter(lambda (x): len(x) > 1) \
                             .collect()
        # Create a N dimensional vector per document using the hashing trick
        fileHash = tf_idf.map(lambda (f, wl): (hashVector(f,wl,100))) \
                         .repartition(8)
    # Loop Through Subjects to build models
    for i in np.arange(0,10):
        print '\nDealing with Subject #%i' % (i+1)
        subject_lists = sc.pickleFile(directory[10] + str(i) + '/') \
                          .map(lambda x: x[1]) \
                          .collect()
        # Naive Bayes, Decision Tree, Logistic Regression Testing
        for j in np.arange(1,4):
            # Make a Copy of the Hashed File
            modelSet = fileHash
            # 3 Machine Learning Algorithms to be used:
            mlA = ['Naive Bayes','Decision Tree Analysis','Logistic Regression']
            print('Subject: %i - Model: %s') % (i,mlA[j-1])
            # Loop through the Sets of Data
            for k in np.arange(1,11):
                # Create List for filtering the Data
                trainingList, testList, validationList = getSetList(ebookNumbers)
                # Print to show Completion
                print 'Cross Validation Fold: %i Complete' % (k)     
    # Stop Watch
    modelTime = time() - modelTime
    print('\n############ Processing Completed ##############')
    print('################################################\n')    
    ######################################################
    
    print('################################################')
    print('##### Display Overall Time and Statistics ######')
    print('Time to Process Word Freq Pickles: %.3f Seconds') % (WordFreq_Time)
    if len(fileEbook) > 0:
        print('Average Time to Process Files:  %.3f Seconds') % (np.mean(np.array([i[4] for i in fileEbook])))
    print('Time to Process IDF Pickle:     %.3f Seconds') % (IDF_Time)
    print('Time to Process TF.IDF Pickle:  %.3f Seconds') % (TFIDF_Time)
    print('Time to Process XML Files:      %.3f Seconds' % (XML_Time))
    print('Time to Test and Build Models:  %.3f Seconds' % (modelTime))
    print('Total Run Time:                 %.3f Seconds') % (time() - start_time_overall)
    print('################################################')
    ######################################################
    # Disconnect from Spark
    sc.stop()
########################################################
