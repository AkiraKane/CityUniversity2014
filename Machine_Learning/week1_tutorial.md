**Rules of Probability**

*Sum Rule*: `p(X) = sum(p(X,Y))`

*Product Rule*: `p(X,Y) = p(Y|X)p(X)`

*Bayes' Theorem*: `p(Y|X) = p(X|Y)p(Y) / p(X)`

`p(A|B) = p(B|A)p(B)/sum(p(B|A))p(A)`

**Example from Lecture Notes:**

A test for salmonella is made available to chicken farmers. The test will correctly show a positive result for salmonella 95% of the time. However, the test also shows a positive result 15% of the time in salmonella free chickens. 10% of chickens have salmonella.

*Question:* If a chicken tests positive, what is the Probability that it has salmonella?

* Let A be the presence of salmonella
* Let B be a positive test
* `P(A|B)` = ?
    * the Probability that a chicken has salmonella given a positive test

* `p(A) = 0.1`
* `p(B|A) = 0.95`
* `p(B|~A) = 0.15`

*Solution:* Using the sum rule:

`p(B) = [ p(B|A)p(A) ] + [ p(B|~A)p(~A)]`

Substitue `p(B)` into the Bayes' Theorem

`p(A|B) = (0.95*0.1)/((0.95*0.1)+(0.15*0.9))`

`p(A|B) = 0.413 = 41.3%` that the chicken has salmonella

**Tutorial 1**

*Question 1*

Suppose we know nothing about coins except that each tossing event produces heads with some
unknown probability p or tails with probability 1-p. Your model of a coin has one parameter, p. You
observe 100 tosses and there are 53 heads. What is p? How about if you only tossed the coin once and
got heads? Is it reasonable to give a single answer if we don’t have much data?

`The value of p given the data that is available is 0.53. Over time, ie if there were a significantly large number of coin tosses, you would expect that if the coin was not loaded for the probability of p to be 0.5.`

*Question 2*

A drugs manufacturer claims that its roadside drug test will detect the presence of cannabis in the blood
(i.e. show positive for a driver who has smoked cannabis in the last *72* hours) *90%* of the time. However,
the manufacturer admits that *10%* of all cannabis-free drivers also will test positive. A national survey
indicates that *20%* of all drivers have smoked cannabis during the last *72* hours.

* What is the probability that a driver has smoked cannabis in the last *72* hours if they have tested
positive?

* Let `A` = probability of positive test
* Let `B` = probability of smoking cannabis = `0.2`
* `p(A|~B) = 0.1`
* `p(A|B) = 0.9`

*Bayes' Theorem*: `p(Y|X) = p(X|Y)p(Y) / p(X)`


`p(B|A) = (0.9*0.2)/(0.8*0.1 + 0.9*0.2)` = `69.23%`

* New information arrives which indicates that, while the roadside drugs test will now show positive
for a driver who has smoked cannabis *99.9%* of the time, the number of cannabis-free drivers
testing positive has gone up to *20%*. What is the probability that someone smoked cannabis in the
last *72* hours if they have not tested positive?

*Question 3*

X-factor viewers from 3 towns took part in a survey about how they voted: 46% of those surveyed were
from Bury, 38% from Croydon and 16% from Dover. The poll showed that 61% of Bury viewers, 88% of
Croydon viewers and 51% of Dover viewers voted for the eventual winner of X-factor. What is the
probability that a vote for the winner was cast by a viewer from Dover?

* `P(A)` = probaility of voting from Dover = 0.16
* `P(~A)` = probaility of not from Dover = 0.84
* `P(B)` = probability of voting for the winner
* `P(B|A)` = 0.51

*Product Rule*: `p(X,Y) = p(Y|X)p(X)`

* `p(A,B) = 0.51*0.16 = 0.0816`

*Question 4*

You don’t know if your friend George knows about neural networks. You think that he studied Artificial
Intelligence, but are not completely sure; let’s say you’re 50% sure. Maybe he did Computer Science;
you’re only 20% sure of that. What are the chances that George knows what a neural network is? You can
assume that 80% of people who study AI know what a neural network is, 40% of people who study CS
know what a neural network is, and 10% of the rest of the population know what a neural network is.

* `p(A)` = probability of AI = 0.5
* `p(B)` = probability of CS = 0.2
* `p(C|A)` = 0.8
* `p(C|B)` = 0.4

*Product Rule*: `p(X,Y) = p(Y|X)p(X)`

`p(B) = [ p(B|A)p(A) ] + [ p(B|~A)p(~A)]`

*Bayes' Theorem*: `p(Y|X) = p(X|Y)p(Y) / p(X)`

*Question 5*

You have a bag containing 1000 coins. 999 are genuine, but one of them is fake, with a ‘head’ on both
sides. You pick one of the 1000 coins at random, and flip it 10 times. All ten times it falls ‘heads’. What is
the probability that this coin is the fake one?
