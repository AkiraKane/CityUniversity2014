#
# INM 432 Big Data solutions for lab 4
# (C) 2014 Daniel Dixey
# http://spark.apache.org/docs/latest/api/python/pyspark.mllib.classification.NaiveBayesModel-class.html
#

# Import Libraries
import sys
import re
from operator import add
# Import the Python Spark API
from pyspark import SparkContext
# Import Pandas - To save library into a Dataframe for Easier Review Post
# Completetion of Script
import numpy as np
# Import specific Machine Learning Library Modules
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

#  A function to Build the Vocabulary List


def buildVocab(x):
    #  Transform the Data
    wordFile = x.map(lambda x_y: (x_y[1], x_y[0]))
    uniqueWords = wordFile.reduceByKey(lambda x, y: x)
    uniqueWords = uniqueWords.collect()
    # Create a Vocabulary List
    vocabularyList = []
    for (w, f) in uniqueWords:
        vocabularyList = vocabularyList + [w]

    return vocabularyList

# Not perfect the best Function: - 1st Trial towards getting words as
# singular - Tillman


def removePlural(word):
    word = word.lower()
    if word.endswith('s'):
        return word[:-1]
    else:
        return word

# Remove the Directory and File Type from the File Variable


def FileExtract(x):
    File = x[x.rfind('/') + 1:]
    return File[:-4]

# Identification of Spam


def isSpam(x):
    if x.startswith('spmsg'):
        return 1
    else:
        return 0

# Convert a [(word,count), ...] list to word vector using the given
# vocabulary, file f is just passed through


def filewordVectorGen(f, wcl, voc):
    vec = [0] * len(voc)  # initialise vevocabularyListctor of vocabulary size
    for wc in wcl:
        if wc[0] not in voc:  # ignore words that are not in our vocabulary
            continue
        i = voc.index(wc[0])  # get word index
        vec[i] = vec[i] + wc[1]  # add count to index
    return (f, vec)

# Import Stop Words Function


def Stop_Words_Map_Reduce(x):
    # Import English stopwords textfile
    stop_words = sc.textFile(x)
    # Tokenise Words from the Stop words text file
    stop_words = stop_words.flatMap(lambda x: re.split('\,', x))
    return stop_words

# Create RDD Function


def createRDD(x, stopWords):
    # Read text files as RDD as (file,textContent) pairs.
    textFiles = sc.wholeTextFiles(x)
    # Create (filename,word pairs) using the flatMap to break up the lists.
    words = textFiles.flatMap(
        lambda f_x: [
            (f_x[0],
             w) for w in re.split(
                '\W+',
                f_x[1])])
    # Filter out the Stop Words from the RDD
    #words = words.filter(lambda (f,w): w not in stop_words)
    # Create a vocabulary List
    vocabularyList = buildVocab(words)
    # Transform Tockenised words to lower case and singular using remPlural
    # Create ((f,w),1) and reduceByKey to count words per file.
    wordsT = words.map(
        lambda f_x1: (
            (FileExtract(
                f_x1[0]), removePlural(
                f_x1[1])), 1))
    wordsT = wordsT.reduceByKey(add)
    # Reorganise the tuples as (f, [(w,c)]).
    fileWord = wordsT.map(
        lambda fw_c: (
            fw_c[0][0], [
                (fw_c[0][1], fw_c[1])]))  # The [] brackets create lists
    fileWord = fileWord.reduceByKey(add)
    # Maximum Term Frequency by File
    FileMaxFreq = wordsT.map(
        lambda fw_c2: (
            fw_c2[0][0],
            fw_c2[1]))  # The [] brackets create lists
    FileMaxFreq = FileMaxFreq.reduceByKey(max)
    # A New  RDD with words as keys and count the occurances of words per file
    # using map()
    WordsperFile = wordsT.map(lambda fw_c3: (
        fw_c3[0][1], (fw_c3[0][0], fw_c3[1])))
    # (word, nd, [(file, count), (file,count),...........,(file,count)])
    df = WordsperFile.map(lambda f_x4: (f_x4[0], [f_x4[1]]))
    df = df.reduceByKey(add)
    # nD is number of files containing word
    df = df.map(lambda x: (x[0], len(x[1]), x[1]))
    # Return the following Arguments
    return fileWord, vocabularyList, FileMaxFreq, df


# The Main Function
if __name__ == "__main__":
    # Make sure we have all arguments we need.
    if len(sys.argv) != 4:
        # Usage: spark-submit wordcount.py /path_to_text_files
        # /path_to_stopwords/stopwords_en.txt
        print >> sys.stderr, "Python File Usage: Lab Sheet 4: <Training Set Directory> <Stopwords_File> <Test Set Directory>"
        exit(-1)

    # Connect to Spark: Create the Job of this Name on the Server
    sc = SparkContext("local[6]", appName="Big Data Lab 4")

    # Stop Words RDD Generation Function
    stop_words = Stop_Words_Map_Reduce(sys.argv[2])

    # Create a Traing set RDD, Vocabulary RDD, Max File Frequency, Word
    # Document Frequency
    trainingSet_RDD, trainingSet_Vocab, trainingSet_MaxF, trainingSet_wrdF = createRDD(
        sys.argv[1], stop_words)
    testSet_RDD, testSet_Vocab, testSet_MaxF, testSet_wrdF = createRDD(sys.argv[
                                                                       3], stop_words)

    # Using the File Name ID if Spam: 1 if Spam - 0 if Otherwise
    TS_SpamHam = trainingSet_RDD.map(lambda f_wc: (isSpam(f_wc[0]), f_wc[1]))
    t_SpamHam = testSet_RDD.map(lambda f_wc5: (isSpam(f_wc5[0]), f_wc5[1]))

    # Convert RDD to a file and word Vector in preparation for the Naive Bayes
    # Modelling
    TrainingSet = TS_SpamHam.map(
        lambda f_wc6: filewordVectorGen(
            f_wc6[0], f_wc6[1], trainingSet_Vocab))
    testSet = t_SpamHam.map(
        lambda f_wc7: filewordVectorGen(
            f_wc7[0], f_wc7[1], trainingSet_Vocab))

    # Labelled Point Setup - TS = Training Set, t = Test Set
    TS_labelledPoints = TrainingSet.map(
        lambda x_l: LabeledPoint(x_l[0], x_l[1]))

    # Train a naive Bayes model
    model = NaiveBayes.train(TS_labelledPoints, 1.0)

    # Prediction using the Trained Model
    evalModel = testSet.map(
        lambda f_vec: (
            float(
                f_vec[0]), model.predict(
                np.array(
                    f_vec[1]))))

    # Compute training error
    trainErr = evalModel.filter(lambda v_p: v_p[0] != v_p[
                                1]).count() / float(t_SpamHam.count())
    print('Training Error = %.5f' % (trainErr * 100))
    print('Accuracy = %.5f' % ((1 - trainErr) * 100))

    sc.stop()  # Disconnect from Spark
