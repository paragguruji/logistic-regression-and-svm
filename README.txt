=============================================================

Comparison of Logistic Regression (LR), 
Support Vector Machine (SVM) and Naive Bayes Classifier on
the Text Classification task over yelp reviews data
=============================================================

Purdue University Spring 2017 CS 573 Data Mining Homework 3
=============================================================

Author: Parag Guruji
Email: pguruji@purdue.edu
=============================================================
 
Python Version 2.7
=============================================================

Directory structure:

parag_guruji/
	|---__init__.py
	|---hw3.py
	|---analysis.pdf
	|---README.txt
	|---requirements.txt
	|---output/ (optional)
		|---comparison_f1.csv
		|---comparison_f1.png
		|---comparison_f2.csv
		|---comparison_f2.png

=============================================================

usage: hw3.py [-h] [-f feature_max_val] [-t ttest ttest]
              [-v ttest_feature_values ttest_feature_values] [-e evaluation]
              trainingDataFilename testDataFilename modelIdx

CS 573 Data Mining HW3 Comparison of LR, SVM and NBC Implementations

positional arguments:
  trainingDataFilename  file-path of training set
  testDataFilename      file-path of testing dataset
  modelIdx              choice of model: 1: Logistic Regression and 2: Support
                        Vector Machine

optional arguments:
  -h, --help            show this help message and exit
  -f feature_max_val, --feature_max_val feature_max_val
                        maximum value a feature can take. i.e. every feature
                        can take a value in {0, ..., feature_max_val}
                        (default: 1)
  -t ttest ttest, --ttest ttest ttest
                        2 Model indices for ttest from stored results. 1: LR,
                        2: SVM, 3: NBC (default: None)
  -v ttest_feature_values ttest_feature_values, --ttest_feature_values ttest_feature_values ttest_feature_values
                        2 Filepaths of results for feature values [0, 1] and
                        for [0, 1, 2] (default: None)
  -e evaluation, --evaluation evaluation
                        file-path of whole data set to be used for performance
                        evaluation as specified in Q3 & Q4. The output files
                        are stored in a subdirectory of current working
                        directory named output (created if doesn't exist)
                        (default: None)
