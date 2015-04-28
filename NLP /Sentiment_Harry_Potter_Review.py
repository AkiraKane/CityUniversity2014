# Daniel Dixey
# Datagenic - Interest
# 28/4/15

# Import Modules
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer

# Demo Text
text = '''
Harry Potter and the Philosopher's stone is a book about a boy called Harry and when he is a baby something terrible happened to his parents. This very evil wizard called Voldemort killed his mum and dad however he tried to kill Harry but somehow he could not. Therefore Harry had to go to live with his aunt and uncle. Eleven years later he had a letter saying he is invited to go to Hogwarts. Harry travelled there on a scarlet red steam engine. At Hogwarts Harry Ron and Hermione, Harry's friends, were caught out of school and their punishment was to collect unicorn blood from the dark woods. Then Harry and his friends go on a big adventure!\n\n
I think the book is very exciting! My favourite part is when Harry and his friends go on a very exciting adventure!\n\n
I think this book is suitable for eight and above. Eleven out of eleven people from Bancffosfelen school said they loved the book! My mark out of ten is nine!\n\n
'''

# Process the Text using the NLTK through Textblob
blob = TextBlob(text)

# Check the Text has been imported correctly
print(text)

# Iterate Through Each Sentence and calculate the Sentiment
for sentence in blob.sentences:
    print(sentence)
    print('Using Pattern Analyser')
    print(TextBlob(sentence.string, 
             analyzer=PatternAnalyzer()).sentiment[0])
    print('Using Naive Bayes')
    print(TextBlob(sentence.string, 
             analyzer=NaiveBayesAnalyzer()).sentiment[0])

# Translate Text
print(blob.translate(to="fr") )  # 'La amenaza titular de The Blob...'
