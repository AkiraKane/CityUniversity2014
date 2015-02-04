#
# INM 432 Big Data solutions for lab 5
# Credit to Nikolay Manchev for the Tidy version
# (C) 2014 Daniel Dixey
#

# Import addtional Modules into Spark
import numpy as np
import re
from operator import add
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.classification import NaiveBayes
from time import time
from pyspark.mllib.tree import DecisionTree
from collections import defaultdict

# A simple attempt to from plural words
def rem_plural(word):
    word = word.rstrip('s')
    return word

# Extract the file and words from each Message
def file_and_word(f,x):
    ret_val = [[f[5:], rem_plural(word)] for word in re.split('\W+',x.lower()) if len(word) > 0]
    return ret_val

# Creating a hashed vector  
def f_wfVector(f, words, N=1000):
    vec = [0]*N
    for word in words:
        h = hash(word)
        vec[h % N] += 1
    return (f,vec)

# Spam of Ham Identification Function
def spam_or_ham(file_name):
    if 'spmsg' in file_name:
        return 1
    else:
        return 0
     
# Use Mapreduce to pre process the RDD for Prediction
def process_fold(rdd, stop_words):
    # Generate the [(word,count), ...] list per file
    words = rdd.flatMap(lambda (f,x): file_and_word(f,x))
    # Filter out English stop words
    words = words.filter(lambda x: x[1] not in stop_words)
    words = words.map(lambda (f,x): ((f,rem_plural(x)),1))
    words = words.reduceByKey(add)
    words_per_file = words.map(lambda (f,x): (f[0],[(f[1],x)]))
    words_per_file = words_per_file.reduceByKey(add)
    # test whether the file is spam
    words_per_file = words_per_file.map(lambda (f,x): (spam_or_ham(f),x))
    return words_per_file
    
def merge_rdds(rdds):
    if len(rdds) == 1:
        return rdds
    elif len(rdds) > 1:
        for i in range(1, len(rdds)):
            rdds[0] = rdds[0].union(rdds[i])
        return rdds[0]
    return None        
    
if __name__ == "__main__":
    start_time = time()
    
    sc = SparkContext(appName = "wordcount")
    
    files_path = '/home/dan/Spark_Files/Spam/bare/part'
    english_stopwords_file = '/home/dan/Spark_Files/Books/stopwords_en.txt'
    
    # Load a list of English stopwords from the provided file stopwords en.txt. The
    # words are separated by commas in the file. The file should be added as an additional
    # argument to spark submit. Tokenise the file content, store it in a Python list...
    stop_words = sc.textFile(english_stopwords_file)
    stop_words = stop_words.flatMap(lambda x: re.split(',',x))
    stop_words = stop_words.collect()
    
    # Load and process individual folds
    rdds = []
    for i in range(1,11):
        fold_path = files_path + str(i)
        rdd = sc.wholeTextFiles(fold_path)
        rdd = process_fold(rdd, stop_words)
        print 'Processed fold %i, RDD contains %i elements' % (i, rdd.count())
        rdds.insert(i, rdd)
        
    # Run cross validation
    resultsTable = np.zeros(shape=(len(rdds),7))
    for i in range(0, len(rdds)):
        print '------------------------'
        print 'Test RDD is RDD[%i]' % i
        print 'Merging training RDDs rdds[:%i] to rdds[%i:]' % (i, i+1)
        test_rdd = rdds[i]
        train_rdds = rdds[:i] + rdds[(i+1):]
        train_rdds = merge_rdds(train_rdds)
        print 'Merged training set contains %i elements' % train_rdds.count()
        
        
        # Build vocabulary and frequency vector
        freq_vect = train_rdds.map(lambda(f,x): f_wfVector(f,x))
        # Create an RDD of LabeledPoint objects
        lpRDD = freq_vect.map(lambda (f,x): LabeledPoint(f,x)).cache()
        
        ################################
        # Naive Bayes - Machine Learning
        ################################
        
        # train the NaiveBayes and save the model as a variable nbModel
        nbModel = NaiveBayes.train(lpRDD, 1.0)
    
        # Generate a Test vector
        predict_vect = test_rdd.map(lambda(f,x): f_wfVector(f,x))
        predict_vect_lb = predict_vect.map(lambda (f,x): LabeledPoint(f,x)).cache()
        
        # Map and Reduce to get Predictions and Results
        resultsTest = predict_vect.map(lambda (l,v):  ((l,nbModel.predict(v)),1)).reduceByKey(add)        
        resultMap = resultsTest.collectAsMap()
        
        # Prepare for Evalulation
        nFiles = predict_vect.count()
        resultMap = defaultdict(lambda :0,resultMap)
        truePos = resultMap[(1,1,)]; falsePos = resultMap[(0,1,)]
        trueNeg = resultMap[(0,0,)]; falseNeg = resultMap[(1,0,)]
        
        # Print Output to an Array
        resultsTable[i,0] = float(truePos+trueNeg)/nFiles
        resultsTable[i,1] = float(truePos)/(truePos+falseNeg)
        resultsTable[i,2] = float(truePos)/(truePos+falsePos)
        resultsTable[i,3] = float(trueNeg)/(trueNeg+falsePos)
        
        ###########################################
        # Decision Tree Analysis - Machine Learning
        ###########################################
        Decision_Tree = DecisionTree.trainClassifier(lpRDD, numClasses=2, categoricalFeaturesInfo={},impurity='gini', maxDepth=5, maxBins=10)
        
        # Evaluate model on training instances and compute training error
        predictions = Decision_Tree.predict(predict_vect_lb.map(lambda x: x.features))
        labelsAndPredictions = predict_vect_lb.map(lambda lp: lp.label).zip(predictions)
        resultsTable[i,4] = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(predict_vect_lb.count())
        resultsTable[i,5] = 1 - resultsTable[i,4]
        
        # Print Decision Tree
        print('Learned Classification Tree Model:')
        print(Decision_Tree)
        
        ########################################
        # Logistic Regression - Machine Learning
        ########################################
        
        # Build the model
        Logistic_Regression = LinearRegressionWithSGD.train(lpRDD, iterations=2)
                
        # Evaluate the model on training data
        valuesAndPreds = predict_vect_lb.map(lambda p: (p.label, Logistic_Regression.predict(p.features)))
        resultsTable[i,6] = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
        
    finish_time = time()
    
    # Display Results
    print '  '
    print '******************************************************'
    print '*****************   Naive Bayes  *********************'
    print 'Average Accuracy of Naive Bayes Analysis: %f' % (np.average(resultsTable[:,0])*100)
    print 'Average Recall of Naive Bayes Analysis: %f' % (np.average(resultsTable[:,1])*100)
    print 'Average Precision of Naive Bayes Analysis: %f' % (np.average(resultsTable[:,2])*100)
    print 'Average Specificity of Naive Bayes Analysis: %f' % (np.average(resultsTable[:,3])*100)
    print '******************************************************'
    print '  '
    print '***************   Decision Tree   ********************'
    print 'Average Accuracy of Decision Tree Analysis: %f' % (np.average(resultsTable[:,4])*100)
    print 'Average Error of Decision Tree Analysis: %f' % (np.average(resultsTable[:,5])*100)
    print '******************************************************'
    print '  '
    print '***************   Logistic Regression  ***************'
    print 'Average MSE of Logistic Regression Analysis: %f' % (np.average(resultsTable[:,6])*100)
    print '******************************************************'
    print '  '
    print 'Total Run Time:', (finish_time - start_time)
    print '******************************************************'
    
sc.stop()
