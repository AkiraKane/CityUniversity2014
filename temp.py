##########################################################
#
# INM 432 - Big Data - Coursework 1
# (C) 2014 Daniel Dixey
#
##########################################################

##########################################################
################# Import Libraries #######################
# Import Spark API
from pyspark import SparkContext 
from pyspark.conf import SparkConf
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
# Import Shutil For Deleting Directories and Files
import shutil
# Import String Module
import string
# Import Mindom Module
from xml.dom import minidom
##########################################################

################ Coursework Question 1A ##################
########### START - Traversing Directories ###############
# Traverse Directories Recursively
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
    print('Directory = %s') % (directory)
    print("Total Size is {0} bytes".format(fileSize))
    print('Number of Files = %i') % (len(fileList))
    print('Total Number of Folders = %i') % (folderCount)
    print('################################################\n')
    return fileList
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
                .filter(lambda (x,y): y<(footerLine))         
    # Extract the list of Word Frequency pairs per file (as an RDD)
    text = text.flatMap(lambda (x,y): (re.split('\W+',x))) \
                .map(lambda x: (str(x),1)) \
                .filter(lambda x: (len(x) > 0)) \
                .reduceByKey(add)
    # Number of unique workds in RDD
    no_words = text.keys().count()    
    # Calculate Term Frequencies
    term_freq = text.map(lambda (x,y): (x,[filename,float(y/no_words)]))
    # Stop Clock
    fin = time()                        
    # Log File Details - For Analysis
    fileEbook += [[filename, Ebook[0], headerLine, footerLine, fin - start]]
    # Delete Text Object - Attempt to Save Space
    del(text)    
    return term_freq
########## END - Removing Header & Footer ##############
########################################################

################## Question 1F #########################
######## START - Calculate the TF.IDF values ###########
# Process Files IDF, TF.IDF and Hashed Vector
def processFiles(term_freq_union):
    # Calculate IDF Values
    idf = term_freq_union.map(lambda (x,fc): (x,[fc])) \
                            .reduceByKey(add) \
                            .map(lambda (x,y): (x,len(y),y)) \
                            .map(lambda (x,y,z): (x,float(np.log(numFiles/y)),z))
    # Save IDF as Pickle Files
    #idf.saveAsPickleFile('/home/dan/Desktop/IMN432-CW01/IDF/' + 'IDF')
    # Calculate TF.IDF Values
    tf_idf = idf.map(lambda (x,y,z): ([[fc[0],float(fc[1]*y),x] for fc in z])) \
                .flatMap(lambda x: x) \
                .map(lambda x: (x[0],([x[2]]))) \
                .reduceByKey(add)
    #tf_idf.saveAsPickleFile('/home/dan/Desktop/IMN432-CW01/TF_IDF/' + 'TF_IDF')
    # Create a 10000 dimensional vector per document using the hashing trick
    fileHash = tf_idf.map(lambda (x,y): (x,hashedVec(y)))
    # Save Hash Vectors - (File [Hash])
    #fileHash.saveAsPickleFile('/home/dan/Desktop/IMN432-CW01/hashVectors/' + 'hashVectors')
    # Return RDDs for further Processing
    return idf, tf_idf, fileHash
# Creating a Hashed Vector
def hashedVec(words, N=10000):
    vec = [0]*N
    for word in words:
        h = hash(str(word))
        vec[h % N] += 1
    return (vec)
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

################## Other Functions #####################
# Extract the File Name - to use as the Folder name when saving Pickles
def FileExtract(x):
    File =  x[x.rfind('/')+1:]   
    return File[:-4]
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
    # Get Hierarchy of Data Structure
    directory = '/home/dan/Desktop/IMN432-CW01/text-part/'
    allFiles = getFileList(directory)
    numFiles = len(allFiles)
    # Create Spark Job Name
    config = SparkConf().setMaster("local[*]")
    sc = SparkContext(conf=config, appName="ACKF415-Coursework-1")
    # Create a File Details List
    fileEbook = []
    ######################################################
    ############### Process DatanumFiles Start ###################
    print('################################################')
    print('###### Process Files > Word Freq to Pickle #####\n')
    # Pickled Word Frequencies
    pickleWordF = getFileList('/home/dan/Desktop/IMN432-CW01/hashVectors/')
    # Check If Pickles Exist
    if len(pickleWordF) < 1:
        print 'Creating Work Freq Pickles and RDDs \n'
        # Loop Through Each of the Files to Extract the Required Parts    
        for i in np.arange(0,3): # Change to specify the number of Files to Process
        ######################################################        
            # Process the Files and return the Individual Document Frequency
            term_freq = processFile(allFiles[i], i, fileEbook)    
            # Combine all the IDF into one RDD - Skip first Run
            if i == 0:
                term_freq_union = term_freq    
            # Union all the Data togeather
            else:
                term_freq_union = term_freq_union.union(term_freq) 
                del(term_freq)
        # Save Pickles
        #term_freq_union.saveAsPickleFile('/home/dan/Desktop/IMN432-CW01/Word_Freq/'+ 'Word_Freq')
    else:
        print 'Using Pre-Pickled Files \n'
    ######################################################
    print('############ Processing Completed ##############')
    print('################################################\n')
    
    print('################################################')
    print('####### Word Freq to IDF and TF.IDF RDDs #######\n')
    ######################################################
    # Start the Timer
    idf_tdf_pickle_time = time()   
    # Check if Pickles have already been created
    pickleIDF = getFileList('/home/dan/Desktop/IMN432-CW01/IDF/')
    # Create Pickles if not already created
    if len(pickleIDF) < 1: 
        # Print Statement
        print 'Creating Pickles and RDDs \n'
        # Calculate the Number of Documents with the Term in it
        idf, tf_idf, fileHash = processFiles(term_freq_union)
    else:
        print 'Using Pre-Pickled Files \n'
    # End Timer fot this phase
    idf_tdf_pickle_time = time() - idf_tdf_pickle_time
    ######################################################   
    print('############ Processing Completed ##############')
    print('################################################\n')
    ######################################################
    
    print('################################################')
    print('############## Process XML Files ###############\n')
    # Start Timer    
    XML_Time = time()
    # Check Pickles have been created
    pickleXML = getFileList('/home/dan/Desktop/IMN432-CW01/processXML/')
    # Create Pickles if not already created
    if len(pickleXML) < 1:
        print 'Creating XML Pickles and RDDs \n'
        file_loc = '/home/dan/Desktop/IMN432-CW01/meta/'
        files = getFileList(file_loc)
        # Loop through Files
        table = []
        # Extract Data from XML Files
        table = xmlExtract(files, table)
        # Process XML Files
        subjects, top_10 = processXML(table)
    else:
        print 'Using Pre-Pickled Files'
    # End Timer fot this phase
    XML_Time = time() - XML_Time
    print('\n############ Processing Completed ##############')
    print('################################################\n')    
    ######################################################
    
    print('################################################')
    print('######### Testing and Building Models ##########\n')    
    # Start Stopwatch    
    model = time()
    # Loop Through Subjects to build models
    for i in np.arange(0,10):
        print 'Dealing with Subject #%i' % (i+1)
        subject_lists = sc.pickleFile('/home/dan/Desktop/IMN432-CW01/processXML/Subject' + str(i) + '/') \
                    .map(lambda x: x[1])
        for j in np.arange(1,4): # Naive Bayes, Decision Tree, Logistic Regression
            if j == 1:
                mlA = 'Naive Bayes'
            elif j == 2:
                mlA = 'Decision Tree Analysis'
            else:
                mlA = 'Logistic Regression'
            print('Subject: %i - Model: %s') % (i,mlA)
            for k in np.arange(1,11):
                print 'Cross Validation Fold: %i Complete' % (k)
    # Stop Watch
    model = time() - model
    print('\n############ Processing Completed ##############')
    print('################################################\n')    
    ######################################################
    
    print('################################################')
    print('##### Display Overall Time and Statistics ######')
    if len(fileEbook) > 0:
        print('Average Time to Process Files:  %.3f Seconds' % (np.mean(np.array([i[4] for i in fileEbook]))))
    print('Time to [IDF, TF.IDF, Hashing]: %.3f Seconds' % (idf_tdf_pickle_time))   
    print('Time to Process XML Files:      %.3f Seconds' % (XML_Time))
    print('Time to Test and Build Models:  %.3f Seconds' % (model))
    print('Total Run Time:                 %.3f Seconds') % (time() - start_time_overall)
    print('################################################')
    ######################################################
    # Disconnect from Spark
    sc.stop()
########################################################
