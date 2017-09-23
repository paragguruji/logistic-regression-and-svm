# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:38:27 2017

@author: Parag
"""

import argparse
import numpy as np
from codecs import open as codecs_open
from re import sub as re_sub
from operator import itemgetter
from random import sample, shuffle
from sys import stdout
from time import time
from os import path, makedirs
import csv
from matplotlib import pyplot
from collections import Counter
from scipy.stats import ttest_ind_from_stats

MODELS = ["LR", "SVM", "NBC"]
ALPHA_SMOOTHING = 1
ALPHA_TTEST = 0.05
ETA_LR = 0.01
ETA_SVM = 0.5
LAMBDA = 0.01
CLASS_LABELS = [0, 1]
DEFAULT_STOPWORD_COUNT = 1
DEFAULT_FEATURE_COUNT = 4000
DEFAULT_TRAINING_CYCLES = 100
K = 10
GIVEN_TSS = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]


def timing(f):
    """Timer Decorator. Unused in final submission to avoid unwanted printing.
        paste @timing on the line above the def of function to be timed.
    """
    def wrap(*args, **kwargs):
        time1 = time()
        ret = f(*args, **kwargs)
        time2 = time()
        print '%s function took %0.3f ms' % \
              (f.func_name, (time2 - time1) * 1000.0)
        return ret
    return wrap


def sigmoid(x):
    """Compute sigmoid
    """
    return 1.0 / (1.0 + np.exp(-x))


def euclidean(a, b):
    """Returns Euclidean distance between 2 vectors a and b
    """
    return np.sqrt(sum([(ai - bi)**2 for ai, bi in zip(a, b)]))


def get_class_labels(model):
    """Return appropriate class labels for given model
    """
    if model in ["LR", "NBC"]:
        return {"positive": 1, "negative": 0}
    if model == "SVM":
        return {"positive": 1, "negative": -1}


def read_tsv_file(file_name):
    """reads the given tab separated data file. If a line has only review text\
        (missing review_id and/or class_label, then appends it to the text of \
        previous valid review)
    """
    data = {}
    last_id = -1
    infp = codecs_open(file_name, "r", encoding='utf-8')
    for line in infp:
        line = [l.strip() for l in line.split('\t')]
        if len(line) != 3:
            if last_id >= 0:
                data[last_id]['review'] += " " + line[0].strip()
            else:
                raise ValueError
        else:
            data[int(line[0])] = {'id': int(line[0]),
                                  'class': int(line[1]),
                                  'review': line[2]}
            last_id = int(line[0])
    return data


def preprocess(data_set):
    """Preprocesses each review text in given dataset as following:
        1. Convert everything to lowercase
        2. Remove all characters except alphanumeric characters and whitespaces
        3. Split words on whitespaces
        4. Keep unique words of review in a set, add it to the record dict as \
            a value with key 'bow', discard original text entry from record
    """
    for i in data_set.keys():
        data_set[i]['bow'] = \
            Counter(re_sub(r'[^\w'+' '+']',
                           '',
                           data_set[i]['review'].lower()).split())
    return data_set


def build_ordered_vocabulary(preprocessed_training_set):
    """Returns a list of unique words sorted in descending order of their \
        frequency. (frequency: no. of records containing that word)
        Tie Resolution: words with equal frequency are randomly shuffled \
            within the range of ranks their respective frequency had.
    """
    vocab = {}
    ordered_vocab = []
    for i in preprocessed_training_set:
        for w in preprocessed_training_set[i]['bow']:
            vocab[w] = 1 if w not in vocab else vocab[w] + 1

    count_words_pairs = sorted([(v, [k for k in vocab if vocab[k] == v])
                                for v in set(vocab.values())],
                               key=itemgetter(0),
                               reverse=True)
    for pair in count_words_pairs:
        ordered_vocab.extend(sample(pair[1], len(pair[1])))
    return ordered_vocab


def build_feature_matrix(preprocessed_data_set, features, feature_max_val=1):
    """Returns 2D array of size eqal to that of <preprocessed_data> \
        where each element is an array of size 2 + size of <features> and \
        format: [<feature_vector>, class_label, review_id]
    """
    return [[min(preprocessed_data_set[i]['bow'][feature], feature_max_val) +
             int(feature == "BIAS")
             for feature in features] +
            [preprocessed_data_set[i]['class'], -1]
            for i in preprocessed_data_set]


def lr_train(x, weights_count):
    """Learn LR from train matrix x
    """
    w = [0.0] * weights_count
    for _ in xrange(DEFAULT_TRAINING_CYCLES):
        for i in xrange(len(x)):
            wx = sum([w[j] * x[i][j] for j in range(weights_count)])
            x[i][-1] = sigmoid(wx)
        w_old = [wj for wj in w]
        for j in xrange(weights_count):
            w[j] = w[j] + \
                   ETA_LR * (sum([xi[j] * (xi[-2] - xi[-1]) for xi in x]) -
                             LAMBDA * w[j])
        if 0.000001 > euclidean(w, w_old):
            break
    return w


def lr_classify(x, w):
    """Apply learned LR w on test matrix x
    """
    for i in xrange(len(x)):
        wx = sum([w[j] * x[i][j] for j in range(len(w))])
        x[i][-1] = int(sigmoid(wx) > 0.5)


def svm_train(x, weights_count):
    """Learn SVM from train matrix x
    """
    def class_label(b):
        return -1*int(not(b)) + b
    N = len(x)
    ETA_BY_N = ETA_SVM / float(N)
    w = [0.0] * weights_count
    is_misclassified = [False] * N
    for _ in xrange(DEFAULT_TRAINING_CYCLES):
        for i in xrange(len(x)):
            x[i][-1] = sum([w[j] * x[i][j] for j in range(weights_count)])
            is_misclassified[i] = bool(class_label(x[i][-2]) * x[i][-1] < 1)
        w_old = [wj for wj in w]
        for j in xrange(weights_count):
            w[j] = w[j] - ETA_BY_N * \
                          sum([LAMBDA * w[j] -
                               (class_label(x[i][-2]) * x[i][j]
                               if is_misclassified[i] else 0)
                               for i in range(N)])
        if 0.000001 > euclidean(w, w_old):
            break
    return w


def svm_classify(x, w):
    """Apply Learned SVM w to test matrix x
    """
    for i in xrange(len(x)):
        wx = sum([w[j] * x[i][j] for j in range(len(w))])
        x[i][-1] = int(wx / abs(wx))


def laplace_smoothing(numerator, denominator, d=2):
    """Returns a Laplace smoothed value for given numerator & denominator terms
    """
    return float(numerator + ALPHA_SMOOTHING) /\
        float(denominator + ALPHA_SMOOTHING*d)


def nbc_train(x, feature_count, feature_max_val=1):
    """Computes nbc parameters: PRIOR and CPDs
    """
    class_labels = get_class_labels("NBC").values()
    params = {'prior': {}}
    prior_numerator = {}
    prior_denominator = float(len(x))
    for k in class_labels:
        prior_numerator[k] = len([feature_vector
                                  for feature_vector in x
                                  if feature_vector[-2] == k])
        params['prior'][k] = prior_numerator[k] / prior_denominator

    params['cpd'] = {}
    for i in range(feature_count):
        params['cpd'][i] = {}
        for j in range(feature_max_val + 1):
            params['cpd'][i][j] = {}
            for k in class_labels:
                """cpd(Xi=j | Y=k)
                """
                params['cpd'][i][j][k] = \
                    laplace_smoothing(len([v for v in x if v[i] == j and
                                           v[-2] == k]), prior_numerator[k])
    return params


def nbc_classify(x, w):
    """Returns result matrix generated by applying the NB classification to \
        test data
    """
    class_labels = get_class_labels("NBC").values()
    for example in x:
        prob = {}
        for k in class_labels:
            prob[k] = \
                reduce((lambda a, b: a * b),
                       [w['cpd'][i][example[i]][k]
                        for i in w['cpd'].keys()])
        example[-1] = max(prob.iteritems(), key=itemgetter(1))[0]


def evaluate_zero_one_loss(test_matrix):
    """Returns zero-one loss from given result_matrix
    """
    return sum([int(example[-2] != int(example[-1]) > 0)
                for example in test_matrix]) / float(len(test_matrix))


def main(**kwargs):
    """Controller function for step-wise execution of learning and application\
        of chosen model - LR OR SVM

        :Returns: (double) zero-one loss
        :kwargs:
            train_set:  training dataset dict
            test_set:   testing dataset dict
            model: {LR, SVM, NBC}
            feature_max_val: maximum value a feature can take (min value 0)
            console_print:  Flag: when True, prints zero-one loss
    """
    preprocessed_training_set = preprocess(kwargs['train_set'])
    ranked_vocabulary = build_ordered_vocabulary(preprocessed_training_set)
    features = sorted(ranked_vocabulary[DEFAULT_STOPWORD_COUNT:
                                        DEFAULT_STOPWORD_COUNT +
                                        DEFAULT_FEATURE_COUNT])
    if kwargs['model'] in ["LR", "SVM"]:
        features = ['BIAS'] + features
    train_matrix = build_feature_matrix(preprocessed_training_set,
                                        features,
                                        kwargs['feature_max_val'])
    preprocessed_testing_set = preprocess(kwargs['test_set'])
    test_matrix = build_feature_matrix(preprocessed_testing_set,
                                       features,
                                       kwargs['feature_max_val'])
    if kwargs['model'] == "LR":
        lr_weights = lr_train(train_matrix, len(features))
        lr_classify(test_matrix, lr_weights)
    elif kwargs['model'] == "SVM":
        svm_weights = svm_train(train_matrix, len(features))
        svm_classify(test_matrix, svm_weights)
    elif kwargs['model'] == "NBC":
        nbc_params = nbc_train(train_matrix,
                               len(features),
                               kwargs['feature_max_val'])
        nbc_classify(test_matrix, nbc_params)
    performance = evaluate_zero_one_loss(test_matrix)
    if 'console_print' in kwargs and kwargs['console_print']:
        print "ZERO-ONE-LOSS-" + kwargs['model'], performance
    stdout.flush()
    return performance


def experiment(filename, feature_max_val=1):
    """Run experiments with CV as specified in qurestion document
    """
    if not filename:
        return
    D = read_tsv_file(filename)
    # Generate K disjoint folds with randomization
    random_seq = D.keys()
    shuffle(random_seq)
    S = [random_seq[i:i + len(D)/K] for i in xrange(0, len(D), len(D)/K)]
    record = {}
    for TSS in GIVEN_TSS:
        record[TSS] = {}
        for m in MODELS:
            performances = []
            for i in xrange(K):
                test_set = {k: v for (k, v) in D.iteritems() if k in S[i]}
                SC = sample([idx for idx in random_seq if idx not in S[i]],
                            int(TSS * len(D)))
                train_set = {k: v for (k, v) in D.iteritems() if k in SC}
                performance = main(train_set=train_set,
                                   test_set=test_set,
                                   model=m,
                                   console_print=False,
                                   feature_max_val=feature_max_val)
                performances.append(performance)
                print "TSS: ", TSS, "Model: ", m, "Performance: ", performance
            record[TSS][m + '_mean'] = np.mean(performances)
            record[TSS][m + '_std_err'] = np.std(performances) / np.sqrt(K)
    return record


def draw_plot(headers, record, output_path):
    """Plot learning curve with error bars
    """
    data = np.array(record)
    colors = ['blue', 'green', 'red']
    for i in range(len(MODELS)):
        pyplot.errorbar(x=data[:, 0],
                        y=data[:, i+1],
                        yerr=data[:, i+4],
                        color=colors[i],
                        label=MODELS[i],
                        marker='o')
    pyplot.xlabel("TSS")
    pyplot.ylabel("ZERO-ONE-LOSS")
    pyplot.title('CS 573 Data Mining HW-3: Comparison of LR, SVM & NBC on \
    Text Classification\n\By: Parag Guruji, pguruji@purdue.edu\n \
    Mean ZERO-ONE-LOSS vs TSS with Std. Errors on error-bars\n',
                 loc='center')
    pyplot.xlim(pyplot.xlim()[0],
                pyplot.xlim()[1] + 0)
    pyplot.legend(loc='upper center', title='Legend')
    pyplot.savefig(output_path + '.png', bbox_inches='tight')
    pyplot.show()


def two_sample_ttest_from_summary(m1, e1, n1, m2, e2, n2):
    """Returns P-value of 2-sample t-test on given summary data
        m(1/2): mean(1/2)
        e(1/2): standard error(1/2)
        n(1/2): sample size(1/2)
    """
    s1 = np.sqrt(n1) * e1
    s2 = np.sqrt(n2) * e2
    t2, p2 = ttest_ind_from_stats(m1, s1, n1,
                                  m2, s2, n2,
                                  equal_var=False)
    print "ttest stats: t = %g  p = %g" % (t2, p2)
    return p2


def ttest(filepath, m1=1, m2=2):
    """Perform two sample T-Test for given 2 models
    """
    with open(filepath, 'r') as datafile:
        datareader = csv.reader(datafile, delimiter=',')
        d = []
        for row in datareader:
            d.append(row)
        row = [[float(i) for i in r] for r in d[1:] if float(r[0]) == 0.15][0]
        p = two_sample_ttest_from_summary(row[m1], row[m1 + 3], K,
                                          row[m2], row[m2 + 3], K)
        str_in = "in"
        str_dont = " DO NOT"
        if p < ALPHA_TTEST:
            str_in = ""
        else:
            str_dont = ""
        print "The difference in mean zero-one-loss of " + MODELS[m1 - 1] + \
            " and that of " + MODELS[m2 - 1] + " is statistically " + \
            str_in + "significant at " + str((1 - ALPHA_TTEST)*100) + \
            "% Confidence Level. \n i.e. Both models materially" + str_dont + \
            " perform the same."


def ttest_f(f1, f2):
    """Perform two sample T-Test for each model across the 2 feature-value-sets
    """
    with open(f1, 'r') as datafile1, open(f1, 'r') as datafile2:
        datareader1 = csv.reader(datafile1, delimiter=',')
        d1 = []
        for row in datareader1:
            d1.append(row)
        r1 = [[float(i) for i in r] for r in d1[1:] if float(r[0]) == 0.15][0]
        datareader2 = csv.reader(datafile2, delimiter=',')
        d2 = []
        for row in datareader2:
            d2.append(row)
        r2 = [[float(i) for i in r] for r in d2[1:] if float(r[0]) == 0.15][0]

        p = [-1] * len(MODELS)
        for i in range(1, len(MODELS)+1):
            p[i-1] = two_sample_ttest_from_summary(r1[i], r2[i + 3], K,
                                                   r2[i], r2[i + 3], K)
            str_in = "in"
            str_dont = " DO NOT"
            if p[i-1] < ALPHA_TTEST:
                str_in = ""
            else:
                str_dont = ""
            print "The difference in mean zero-one-loss of " + MODELS[i - 1] +\
                " in results with feature values [0, 1] and [0, 1, 2]" + \
                " is statistically " + str_in + "significant at " + \
                str((1 - ALPHA_TTEST)*100) + \
                "% Confidence Level. \n i.e. The model materially" + \
                str_dont + " perform the same with both feature-value-sets."


if __name__ == "__main__":
    """Process commandline arguments and make calls to appropriate functions
    """
    parser = \
        argparse.ArgumentParser(
                    description='CS 573 Data Mining HW3  Comparison of LR, SVM\
                    and NBC Implementations',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('trainingDataFilename',
                        help='file-path of training set')
    parser.add_argument('testDataFilename',
                        help='file-path of testing dataset')
    parser.add_argument('modelIdx',
                        metavar='modelIdx',
                        type=int,
                        choices=[1, 2, 3],
                        help='choice of model: 1: Logistic Regression \
                        and 2: Support Vector Machine')
    parser.add_argument('-f', '--feature_max_val',
                        metavar='feature_max_val',
                        type=int,
                        default=1,
                        help='maximum value a feature can take. i.e. every \
                        feature can take a value in {0, ..., feature_max_val}')
    parser.add_argument('-t', '--ttest',
                        metavar='ttest',
                        type=int,
                        nargs=2,
                        default=None,
                        help='2 Model indices for ttest from stored results.\
                        1: LR, 2: SVM, 3: NBC')
    parser.add_argument('-v', '--ttest_feature_values',
                        metavar='ttest_feature_values',
                        nargs=2,
                        default=None,
                        help='2 Filepaths of results for feature values \
                        [0, 1] and for [0, 1, 2]')
    parser.add_argument('-e', '--evaluation',
                        metavar='evaluation',
                        default=None,
                        help="file-path of whole data set to be used for \
                        performance evaluation as specified in Q3 & Q4. The \
                        output files are stored in a subdirectory of current \
                        working directory named output (created if doesn't \
                        exist)")
    args = parser.parse_args()
    if args.ttest_feature_values:
        ttest_f(args.ttest_feature_values[0], args.ttest_feature_values[0])
    elif args.ttest:
        ttest(path.join('output',
                        'comparison_f' + str(args.feature_max_val)) + '.csv',
              m1=args.ttest[0], m2=args.ttest[1])
    elif args.evaluation:
        record = experiment(args.evaluation, args.feature_max_val)
        columns = ['TSS'] + [col for col in
                             [m + '_mean' for m in MODELS] +
                             [m + '_std_err' for m in MODELS]]
        rows = [[TSS] + [record[TSS][col] for col in
                [m + '_mean' for m in MODELS] +
                [m + '_std_err' for m in MODELS]] for TSS in record]
        rows.sort(key=itemgetter(0))
        if not path.isdir('output'):
            makedirs('output')
        with open(path.join('output',
                            'comparison_f' + str(args.feature_max_val)) +
                  '.csv', "wb") as f:
            w = csv.writer(f)
            w.writerow(columns)
            print columns
            for row in rows:
                w.writerow(row)
                print row
        draw_plot(columns,
                  rows,
                  path.join('output',
                            'comparison_f' + str(args.feature_max_val)))
        ttest(path.join('output',
                        'comparison_f' + str(args.feature_max_val)) + '.csv')
    else:
        training_set = read_tsv_file(args.trainingDataFilename)
        testing_set = read_tsv_file(args.testDataFilename)
        arguments = {'train_set': training_set,
                     'test_set': testing_set,
                     'model': MODELS[args.modelIdx - 1],
                     'feature_max_val': args.feature_max_val,
                     'console_print': True}
        main(**arguments)
