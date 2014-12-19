###################################################################
# How to Write a Spelling Corrector
# http://norvig.com/spell-correct.html
###################################################################

###################################################################
# Import Modules
import re, collections
from time import time

###################################################################
# Define Functions
def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get).capitalize()

def spelltest(tests, bias=None, verbose=False):
    n, bad, unknown, start = time(), 0, 0, time()
    if bias:
        for target in tests: NWORDS[target] += bias
    for target,wrongs in tests.items():
        for wrong in wrongs.split():
            n += 1
            w = correct(wrong)
            if w!=target:
                bad += 1
                unknown += (target not in NWORDS)
                if verbose:
                    print '%r => %r (%d); expected %r (%d)' % (
                        wrong, w, NWORDS[w], target, NWORDS[target])
    return dict(bad=bad, n=n, bias=bias, pct=float(100. - 100.*bad/n), 
                unknown=unknown, secs=float(time()-start) )
###################################################################
# Create Vector of Words
NWORDS = train(words(file('/home/dan/Spark_Files/Web/big.txt').read()))
# Define the Alphabet
alphabet = 'abcdefghijklmnopqrstuvwxyz'

###################################################################
tests1 = { 'access': 'acess', 'accessing': 'accesing', 'accommodation':
    'accomodation acommodation acomodation', 'account': 'acount'}

tests2 = {'forbidden': 'forbiden', 'decisions': 'deciscions descisions',
    'supposedly': 'supposidly', 'embellishing': 'embelishing'}

print spelltest(tests1)
print spelltest(tests2) ## only do this after everything is debugged

###################################################################
