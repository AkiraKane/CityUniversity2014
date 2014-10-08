#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# import libraries
import sys
import re
from operator import add
from pyspark import SparkContext 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: wordcount <file>"
        exit(-1)
    # Spark Job Name
    sc = SparkContext(appName="PythonWordCount")
    # Import Files
    lines = sc.textFile(sys.argv[1], 1)
            
    # Read Stop Word Files
    stopwords = ['a',	'about',	'above',	'after',	'again',	'against',	'all',	'am',	'an',	'and',	'any',	'are',	'as',	'at',	'be',	'because',	'been',	'before',	'being',	'below',	'between',	'both',	'but',	'by',	'cannot',	'could',	'did',	'do',	'does',	'doing',	'down',	'during',	'each',	'few',	'for',	'from',	'further',	'had',	'has',	'have',	'having',	'he',	'her',	'here',	'hers',	'herself',	'him',	'himself',	'his',	'how',	'i',	'if',	'in',	'into',	'is',	'it',	'its',	'itself',	'me',	'more',	'most',	'my',	'myself',	'no',	'nor',	'not',	'of',	'off',	'on',	'once',	'only',	'or',	'other',	'ought',	'our',	'ours ',	'out',	'over',	'own',	'same',	'she',	'should',	'so',	'some',	'such',	'than',	'that',	'the',	'their',	'theirs',	'them',	'themselves',	'then',	'there',	'these',	'they',	'this',	'those',	'through',	'to',	'too',	'under',	'until',	'up',	'very',	'was',	'we',	'were',	'what',	'when',	'where',	'which',	'while',	'who',	'whom',	'why',	'with',	'would',	'you',	'your',	'yours',	'yourself',	'yourselves']
         
    # Read Input File
    words = lines.flatMap(lambda x: re.split('\W+',x))
    wordsMap = words.map(lambda x: (x.lower(),1))
    freqWords =  wordsMap.reduceByKey(add)\
                  .filter(lambda x: x[0] not in stopwords) \
                  .filter(lambda x: len(x[0]) > 0 and x[1] > 400) 
    output = freqWords.collect()
    # Output visual
    for (word, count) in output:
        print "Word: %s:    Frequency: %i" % (word.encode('utf-8','replace'), count)

    sc.stop()
