#
# INM 432 Big Data solutions for lab 2
#

import sys
import re
from operator import add
from time import sleep

from pyspark import SparkContext

# Task 1, this function converts to lower case and removes trailing 's'
# not perfect, but a 1st step towards getting words as singular
def remPlural( word ):
    word = word.lower()
    if word.endswith('s'):
        return word[:-1]
    else:
        return word

# list of function words
stopwords = ['in','on','of','out','by','from','to','over','under','the','a','when', \
             'where','what','who','whom','you','thou','go','must','i','me','my','myself']

# The main function
if __name__ == "__main__":
    # Make sure we have all arguments we need.
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: lab2 <directory>"
        exit(-1)
    
    # Connect to Spark
    sc = SparkContext(appName="Big Data Lab 2") # give this job a name

    # Task 2a. Loads all files in the given directory
    textFiles = sc.wholeTextFiles(sys.argv[1]) # read text files as RDD with (file,textContent) pairs.
    #output = textFiles.collect() # for testing

    # Task 2b. Split text into words, and create (filename,word pairs). flatMap break up the lists
    words = textFiles.flatMap(lambda (f,x): [(f, w) for w in re.split('\W+',x)])
    #output = words.collect() # for testing

    # this help during development by keeping the data small, and can be useful for text style classification
    words = words.filter(lambda x: x[1] not in stopwords)

    # Task 1. Transform words to lower case and singular using remPlural
    # Task 2c. Create ((f,w),1) and reduceByKey to count words per file.
    words1 = words.map(lambda (f,x): ((f,remPlural(x)),1))
    words1 = words1.reduceByKey(add)
    words1 = words1.filter(lambda x: x[1] > 100)
    #output = words1.collect() # for testing

    #Task 2c. Reorganise the tuples as (f, [(w,c)]).
    words1 = words1.map(lambda (fw,c): (fw[0],[(fw[1],c)])) # The [] brackets create lists
    # ... that can be concatenated by add:
    words1 = words1.reduceByKey(add)
    output = words1.collect() # get the result

    for k in output: # look at the output
        print k

    # Delays Output for 5 Seconds
    sleep(5)
    sc.stop() # Disconnect from Spark
