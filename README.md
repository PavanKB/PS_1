

Two interesting **classification** problems

# DATASETS
What are the datasets and why were they chosen
https://archive.ics.uci.edu/
1. Choose data with many features - high dimensionality
2. Choose data with many instances - large sample size
3. Skewed data - imbalanced classes ?

*large features and sample size*
https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications  

# EDA
To be done on full data
1. Show the imbalance in Y
1. Show the distribution by key features - FacetGrid
1. How they are correlated
1. Any empty values


# Data prep
1. Drop empty - no imputation here
2. One hot encoding
3. Standardisation - MinMaxScaler - # Use minmax scalerr for simplicity, scalability, avoids assumption of distribution and there are no outliers in data

# Training

Split data using stratified K fold. K = 20
Evaluation metric to use is F-beta since this is an imbalance dataset
both False Negatives and False Positives are equally important then we use F1-Score

1. Gridsearch tuning  - to find best tree

DEciosn tree wont work for imbalanced dataset

## Hyperparameter tuning
Atleast 2 for each algorithm
- DT: Pruning, which attribute did you choose to split on?
- Boosting: # of Weak Learners
- NN: Hidden Layer Size (Width, Depth), describe which activation function you chose
- SVM: Kernel Type, atleast two kernel types
- KNN: K

## Model evaluation
Precision, Recall, F1, Accuracy, Confusion Matrix

Use SHAPS to explain the model

1. After getting the best params
    1. Include learning curves for training 
    1. validation accuracy and loss.
    1. ROC/AUC curves for test data.
    1. Confusion matrix

Talk about bias variance trade off for each model

https://github.com/hyperopt/hyperopt
https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0


# Code Setup
1. how to retrieve your code, 
2. how to retrieve your data, 
3. how to retrieve the dependencies (libraries) for running your code, 
4. and how to run your code.
5. Implement command prompt execution of your code.



# Citations!

https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

https://www.freecodecamp.org/news/machine-learning-pipeline/

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

https://medium.com/cuenex/advanced-evaluation-metrics-for-imbalanced-classification-models-ee6f248c90ca

https://seaborn.pydata.org/tutorial/relational.html

https://www.analyticsvidhya.com/blog/2021/07/using-seaborns-facetgrid-based-methods-for-exploratory-data-analysis/

https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/