#
# INM 432 Big Data solutions for lab 3
# (C) 2014 Daniel Dixey
#

import sys
import re
from operator import add
from pyspark import SparkContext
import pandas as pd

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

# The Main Function
if __name__ == "__main__":
    # Make sure we have all arguments we need.
    if len(sys.argv) != 3:
        # Usage: spark-submit wordcount.py /path_to_text_files
        # /path_to_stopwords/stopwords_en.txt
        print >> sys.stderr, "Python File Usage: Lab Sheet 3: <directory> <stopwords_file>"
        exit(-1)

    # Connect to Spark: Create the Job of this Name on the Server
    sc = SparkContext(appName="Big Data Lab 3")

    # Loads all files in the given directory into one RDD
    # Read text files as RDD as (file,textContent) pairs.
    textFiles = sc.wholeTextFiles(sys.argv[1])
    # Import English stopwords textfiles
    stop_words = sc.textFile(sys.argv[2])
    # output = stop_words.collect() # For Testing Purposes

    # Splitting of textfiles into Words.
    # To create (filename,word pairs) using the flatMap to break up the lists.
    words = textFiles.flatMap(
        lambda f_x: [
            (f_x[0],
             w) for w in re.split(
                '\W+',
                f_x[1])])
    # Tokenise Words from the Stop words text file
    stop_words = stop_words.flatMap(lambda x: re.split('\,', x))

    # Filter out the Stop Words from the RDD
    word = words.filter(lambda x: x[1] not in stop_words)

    # Transform Tockenised words to lower case and singular using remPlural
    # Create ((f,w),1) and reduceByKey to count words per file.
    wordsT = words.map(
        lambda f_x1: (
            (FileExtract(
                f_x1[0]), removePlural(
                f_x1[1])), 1))
    wordsT = wordsT.reduceByKey(add)

    # Reorganise the tuples as (f, [(w,c)]).
    # The [] brackets create lists
    words2 = wordsT.map(lambda fw_c: (fw_c[0][0], [(fw_c[0][1], fw_c[1])]))
    words2 = words2.reduceByKey(add)

    # Maximum Term Frequency by File
    FileMaxFreq = wordsT.map(
        lambda fw_c2: (
            fw_c2[0][0],
            fw_c2[1]))  # The [] brackets create lists
    FileMaxFreq = FileMaxFreq.reduceByKey(max)

    # Create a New  RDD with words as keys and count the occurances of words
    # per file using map()
    WordsperFile = wordsT.map(lambda fw_c3: (
        fw_c3[0][1], (fw_c3[0][0], fw_c3[1])))

    # Task 3E: With the output of 4 use map to create tuples of this form:
    # (word, nd, [(file, count), (file,count),...........,(file,count)])
    idf_calc_prep = WordsperFile.map(lambda f_x4: (f_x4[0], [f_x4[1]]))
    idf_calc_prep = idf_calc_prep.reduceByKey(add)
    # nD is number of files containing word
    idf_calc_prep = idf_calc_prep.map(lambda x: (x[0], len(x[1]), x[1]))
    output = idf_calc_prep.collect()

    # Output as a CSV File
    Output = pd.DataFrame(output)
    Output.to_csv('Output.csv', sep=',', index=False)

    sc.stop()  # Disconnect from Spark
