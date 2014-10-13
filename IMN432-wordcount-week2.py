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
import os
import sys
import re
from operator import add
from pyspark import SparkContext 

# define new function
def remPlural(x):
    word = x.lower()
    if word.endswith('s'):
        return [word[:-1],1]
    else:
        return [word,1]

if __name__ == "__main__":
    # Spark Job Name
    sc = SparkContext(appName="PythonWordCount")
    # Import Files
    allFiles = sc.wholeTextFiles("/home/dan/Spark_Files/")
        
    # Read Stop Word Files
    stopwords = ['a',	'about',	'above',	'after',	'again',	'against',	'all',	'am',	'an',	'and',	'any',	'are',	'as',	'at',	'be',	'because',	'been',	'before',	'being',	'below',	'between',	'both',	'but',	'by',	'cannot',	'could',	'did',	'do',	'does',	'doing',	'down',	'during',	'each',	'few',	'for',	'from',	'further',	'had',	'has',	'have',	'having',	'he',	'her',	'here',	'hers',	'herself',	'him',	'himself',	'his',	'how',	'i',	'if',	'in',	'into',	'is',	'it',	'its',	'itself',	'me',	'more',	'most',	'my',	'myself',	'no',	'nor',	'not',	'of',	'off',	'on',	'once',	'only',	'or',	'other',	'ought',	'our',	'ours ',	'out',	'over',	'own',	'same',	'she',	'should',	'so',	'some',	'such',	'than',	'that',	'the',	'their',	'theirs',	'them',	'themselves',	'then',	'there',	'these',	'they',	'this',	'those',	'through',	'to',	'too',	'under',	'until',	'up',	'very',	'was',	'we',	'were',	'what',	'when',	'where',	'which',	'while',	'who',	'whom',	'why',	'with',	'would',	'you',	'your',	'yours',	'yourself',	'yourselves']
    
    # All Files
    Texts = allFiles.flatMap(lambda (f,x): [(f,w) for w in re.split('\W+',x.lower())]) \
          .map(lambda x: (x,1)) \
          .reduceByKey(add) \
          .filter(lambda x: x[1] > 500)
 
    Output = Texts.collect()
    for (word, count) in Output:
        print "Directory & Word: %s:    Frequency: %i" % (word, count)
    
    sc.stop()
