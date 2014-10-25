#
# INM 432 Big Data solutions for lab 4
# (C) 2014 Daniel Dixey
#

# Import Libraries
import sys
import re
from operator import add
# Import the Python Spark API
from pyspark import SparkContext
# Import Pandas - To save library into a Dataframe for Easier Review Post Completetion of Script
#import pandas as pd
#import numpy as np
# Import Machine Learning Library Modules
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

#  A function to Build the Vocabulary List
def buildVocab(x):
    #  Transform the Data
    wordFile = x.map(lambda (x,y): (y,x))
    uniqueWords = wordFile.reduceByKey(lambda x,y: x)
    uniqueWords = uniqueWords.collect()
    # Create a Vocabulary List    
    vocabularyList = []    
    for (w,f) in uniqueWords:
        vocabularyList = vocabularyList + [w]
    
    return vocabularyList

# Not perfect the best Function: - 1st Trial towards getting words as singular - Tillman
def removePlural(word):
    word = word.lower()
    if word.endswith('s'):
        return word[:-1]
    else:
        return word

# Remove the Directory and File Type from the File Variable
def FileExtract(x):
    File =  x[x.rfind('/')+1:]   
    return File[:-4]

# Identification of Spam
def isSpam(x):
    if x.startswith('spmsg'):
        return 1
    else:
        return 0
        
# Convert a [(word,count), ...] list to word vector using the given vocabulary, file f is just passed through
def filewordVectorGen(f, wcl, voc ):
    vec = [0] * len(voc) # initialise vevocabularyListctor of vocabulary size
    for wc in wcl :
        if wc[0] not in voc: # ignore words that are not in our vocabulary
            continue
        i = voc.index(wc[0]) # get word index
        vec[i] = vec[i] + wc[1] # add count to index
    return (f,vec)

# Import Stop Words Function
def Stop_Words_Map_Reduce(x):
    # Import English stopwords textfile
    stop_words = sc.textFile(x)
    # Tokenise Words from the Stop words text file
    stop_words = stop_words.flatMap(lambda x: re.split('\,', x))
    return stop_words

# Create RDD Function
def createRDD(x,stopWords):
    # Read text files as RDD as (file,textContent) pairs.
    textFiles = sc.wholeTextFiles(x)
    # Create (filename,word pairs) using the flatMap to break up the lists.
    words = textFiles.flatMap(lambda (f,x): [(f, w) for w in re.split('\W+',x)])  
    # Filter out the Stop Words from the RDD
    #words = words.filter(lambda (f,w): w not in stop_words)    
    # Create a vocabulary List
    vocabularyList = buildVocab(words)
    # Transform Tockenised words to lower case and singular using remPlural
    # Create ((f,w),1) and reduceByKey to count words per file.
    wordsT = words.map(lambda (f,x): ((FileExtract(f),removePlural(x)),1))
    wordsT = wordsT.reduceByKey(add)
    # Reorganise the tuples as (f, [(w,c)]).
    fileWord = wordsT.map(lambda (fw,c): (fw[0],[(fw[1],c)])) # The [] brackets create lists
    fileWord = fileWord.reduceByKey(add)
    # Maximum Term Frequency by File
    FileMaxFreq = wordsT.map(lambda (fw,c): (fw[0],c)) # The [] brackets create lists
    FileMaxFreq = FileMaxFreq.reduceByKey(max)
    # A New  RDD with words as keys and count the occurances of words per file using map()
    WordsperFile = wordsT.map(lambda (fw,c): (fw[1], (fw[0],c)))
    
    # (word, nd, [(file, count), (file,count),...........,(file,count)])
    df = WordsperFile.map(lambda (f,x): (f,[x]))
    df = df.reduceByKey(add)
    # nD is number of files containing word
    df = df.map(lambda x: (x[0], len(x[1]), x[1]))
    # Return the following Arguments
    return fileWord, vocabularyList, FileMaxFreq, df

    
# The Main Function
if __name__ == "__main__":
    # Make sure we have all arguments we need.
    if len(sys.argv) != 4:
        # Usage: spark-submit wordcount.py /path_to_text_files /path_to_stopwords/stopwords_en.txt
        print >> sys.stderr, "Python File Usage: Lab Sheet 4: <Training Set Directory> <Stopwords_File> <Test Set Directory>"
        exit(-1)

    # Connect to Spark: Create the Job of this Name on the Server
    sc = SparkContext("local[6]",appName="Big Data Lab 4") 
    
    # Stop Words RDD Generation Function
    stop_words = Stop_Words_Map_Reduce(sys.argv[2])
    
    # Create a Traing set RDD, Vocabulary RDD, Max File Frequency, Word Document Frequency
    trainingSet_RDD, trainingSet_Vocab, trainingSet_MaxF, trainingSet_wrdF = createRDD(sys.argv[1],stop_words)
    testSet_RDD, testSet_Vocab, testSet_MaxF, testSet_wrdF = createRDD(sys.argv[3],stop_words) 
    
    # Using the File Name ID if Spam: 1 if Spam - 0 if Otherwise
    TS_SpamHam = trainingSet_RDD.map(lambda (f,wc): (isSpam(f),wc))
    t_SpamHam = testSet_RDD.map(lambda (f,wc): (isSpam(f),wc))
    
    # Convert RDD to a file and word Vector in preparation for the Naive Bayes Modelling
    TrainingSet = TS_SpamHam.map(lambda (f,wc): filewordVectorGen(f,wc,trainingSet_Vocab))
    testSet = t_SpamHam.map(lambda (f,wc): filewordVectorGen(f,wc,trainingSet_Vocab))
    
    # Labelled Point Setup - TS = Training Set, t = Test Set
    TS_labelledPoints = TrainingSet.map(lambda (x,l): LabeledPoint(x,l))
    t_labelledPoints = testSet.map(lambda (x,l): LabeledPoint(x,l))
       
    # Train a naive Bayes model
    model = NaiveBayes.train(TS_labelledPoints, 1.0)

    # Make prediction.
    #prediction = model.predict(myarray)
    
    # Output as a CSV File for Easy Reading!
    #Output = pd.DataFrame(output2)
    #Output.to_csv('Output.csv', sep=',', index=False)
    
    sc.stop() # Disconnect from Spark
