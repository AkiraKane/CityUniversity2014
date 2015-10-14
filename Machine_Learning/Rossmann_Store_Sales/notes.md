### Notes for Machine Learning Coursework

*Overview*

On completing this coursework you should be able to:
1. Specify a Machine Learning (ML) solution to a data analysis problem;
2. Adjust the ML model parameters and explain how ML seeks to solve the problem;
3. Apply, compare and contrast, and critically evaluate two ML models.

Apply, compare and contrast two ML solutions: adjust the ML model parameters (e.g. using Matlab) and explain how ML seeks to solve the problem.

*General Guidelines*

The idea with this coursework is to give you some experience carrying out and presenting a piece of research in machine learning. What we expect to see is a task that you describe clearly, relate to existing work, implement and test on a dataset using two ML models. To do this you will need to write or change code, run it on some data, make some figures, read a few background papers, collect some references, compare and contrast results, and create a poster describing your task, the algorithm(s) you used and the results you’ve obtained. As a rough rule of thumb, spend about a week’s worth of work (spread out over a longer time to allow the computers to do some work in the interim!), and about three days preparing the poster and rehearsing your presentation.

*Specific Requirements*

You are asked to compare two machine learning algorithms in practice, when applied to some data. You may also propose a new algorithm, in which case you still should compare it to one other approach. Select the data and algorithm(s) wisely! Your poster should include at least two figures which graphically illustrate quantitative aspects of your results, such as training/testing error curves, learned parameters, algorithm outputs, etc. It should also include at least 5 references to research papers or book chapters.

**Code**

*Marking*:
1. syntactic correctness (5%)
2. organization and clarity of comments (10%)
3. appropriate use and sophistication of methods (10%).

*Possible Datasets*
* [Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales "Competition") - [Kaggle](https://www.kaggle.com "Kaggle") Competition

*Coding Steps*
* Import data
* Basic Statistics about the data
    * Missing data ([Pandas](http://pandas.pydata.org/))
    * Column Statistics ([Pandas](http://pandas.pydata.org/))
    * Plotting Pertinent Variables ([Seaborn](http://stanford.edu/~mwaskom/software/seaborn/))
    * Basic Statistics
* Preprocessing
    * Removing Data based on previous step
    * [Cross-validation](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_validation)
* Machine Learning Processing
    * [Possible Models](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning "supervised learning models")
    * Optimisation/Model Selection
        * [Bayes Optimization](https://github.com/fmfn/BayesianOptimization) - Preferred
        * [Random/Grid Search](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.grid_search) -Second Choice
        * PCA / Feature Engineering - After Optimization
* Results Metrics
    * [Typical Metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
    * Graphical Representation ([Seaborn](http://stanford.edu/~mwaskom/software/seaborn/))

*Feature Engineering Ideas*

1. Features Created
    * Mean Sales per Month & Year (*t-statistic* validation)
    * Means Sales per Month & Year & Ration to Store (*t-statistic* validation)
    * Mean On/Off Promotion & Ratio (*t-statistic* validation)
    * Mean On/Off School Holiday & Ratio to Store (*t-statistic* validation)
    * Mean On/Off State Holiday & Ratio to Store (*t-statistic* validation)
    * Mean On/Off Weekend & Ratio for Store (*t-statistic* validation)
    * Log Sales
2. Relationships Explored
    * Each Variables to Predictor
    * Feature importance
    * Principal Component Analysis (PCA)

**Poster**

*Marking*:

1. brief description and motivation of the problem (5%)
2. initial analysis of the data set including basic statistics (10%)
3. brief summary of the two ML models with their pros and cons (10%)
4. hypothesis statement (5%)
5. description of choice of training and evaluation methodology (5%)
6. choice of parameters and experimental results (10%)
7. analysis and critical evaluation of results (25%)
8. lessons learned and future work (5%)

*Guide*

* https://www.asp.org/education/EffectivePresentations.pdf

**References/Papers**

*Keywords*

* Gradient Decent
* Cross-validation
* Supervised learning

*References*

1. Snoek, J., Larochelle, H., Adams, R.P., 2012. Practical Bayesian optimization of machine learning algorithms, in: Advances in Neural Information Processing Systems. pp. 2951–2959.
2. Nonlinear Regression Analysis and Its Applications - http://onlinelibrary.wiley.com/book/10.1002/9780470316757
3. Rasmussen, C.E., 2006. Gaussian processes for machine learning. MIT Press.
4. Sun, Z.-L., Choi, T.-M., Au, K.-F., Yu, Y., 2008. Sales forecasting using extreme learning machine with applications in fashion retailing. Decision Support Systems 46, 411 – 419. doi:http://dx.doi.org/10.1016/j.dss.2008.07.009
