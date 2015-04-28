from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

text = '''
Harry Potter and the Philosopher's stone is a book about a boy called Harry and when he is a baby something terrible happened to his parents. This very evil wizard called Voldemort killed his mum and dad however he tried to kill Harry but somehow he could not. Therefore Harry had to go to live with his aunt and uncle. Eleven years later he had a letter saying he is invited to go to Hogwarts. Harry travelled there on a scarlet red steam engine. At Hogwarts Harry Ron and Hermione, Harry's friends, were caught out of school and their punishment was to collect unicorn blood from the dark woods. Then Harry and his friends go on a big adventure!\n\n
I think the book is very exciting! My favourite part is when Harry and his friends go on a very exciting adventure!\n\n
I think this book is suitable for eight and above. Eleven out of eleven people from Bancffosfelen school said they loved the book! My mark out of ten is nine!\n\n
'''

blob = TextBlob(text)

print(text)

for sentence in blob.sentences:
    print(sentence)
    print(TextBlob(sentence.string, 
             analyzer=NaiveBayesAnalyzer()).sentiment[0])
    

print(blob.translate(to="fr") )  # 'La amenaza titular de The Blob...'
