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
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: Lab Sheet 3: <directory> <stopwords_file>"
        exit(-1)
    
    # Connect to Spark
    sc = SparkContext(appName="Big Data Lab 3") # give this job a name

    # Task 2A. Loads all files in the given directory
    textFiles = sc.wholeTextFiles(sys.argv[1]) # read text files as RDD with (file,textContent) pairs.
    # Task 3A: Import English stopwords textfiles
    stop_words = sc.textFile(sys.argv[2])
    #output = stop_words.collect() # for testing
    #print output
    
    # Task 2B. Split text into words, and create (filename,word pairs). flatMap break up the lists
    words = textFiles.flatMap(lambda (f,x): [(f, w) for w in re.split('\W+',x)])
    # Task 3B: Tokenise Words from Textfile
    stop_words = stop_words.flatMap(lambda x: re.split('\,', x))
    stop_words = stop_words.collect() # for testing
    # print stop_words
    
    # this help during development by keeping the data small, and can be useful for text style classification
    words = words.filter(lambda x: x[1] not in stop_words) 
    # output = words.collect() # for testing    
    
    # Task 1. Transform words to lower case and singular using remPlural
    # Task 2C. Create ((f,w),1) and reduceByKey to count words per file.
    words1 = words.map(lambda (f,x): ((f,remPlural(x)),1))
    words1 = words1.reduceByKey(add)
    #words1 = words1.filter(lambda (f,x): x>100)
    #output = words1.collect() # for testing
    
    #Task 2C. Reorganise the tuples as (f, [(w,c)]).
    words12 = words1.map(lambda (fw,c): (fw[0],[(fw[1],c)])) # The [] brackets create lists
    # ... that can be concatenated by add:
    words12 = words12.reduceByKey(add)
    #output = words12.collect()
    
    # Task 3C: Maximum Term Frequency
    wordsMax = words1.map(lambda (fw,c): (fw[0],c)) # The [] brackets create lists
    wordsMax = wordsMax.reduceByKey(max)
    #output = wordsMax.collect()
    
    # Task 3D: Create a New  RDD with words as key and count the occurances of words per file using map()
    WordMap = words1.map(lambda (fw,c): (fw[1],[fw[0],c]))
    # Reduce by to get the lists of term frequencies per file
    WordMap = WordMap.reduceByKey(add)
    #output = WordMap.collect()
    
    # Task 3E: With the output of 4 use map to create tuples of this form:
    # (word, nd, [(file, count), (file,count),...........,(file,count)])
    
    #for k in output: # look at the output
    #    print k

    # Delays Output for 5 Seconds
    #sleep(5)
    sc.stop() # Disconnect from Spark
