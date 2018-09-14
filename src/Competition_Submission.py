print('\nImporting Packages...')


import numpy as np
import pandas as pd
import pickle
import urllib.request

from IPython.core.interactiveshell import InteractiveShell
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

InteractiveShell.ast_node_interactivity = "all"


def split_data_equal(x, y, test_set, train_size=0.75):
    '''
    splits data into train and test sets and standardizes the data
    :param x: data to be split into train/test and standardized
    :param y: labels
    :param test_set: data to be split into train/test and standardized with unknown labels
    :train_size: the portion of the data to be split as training data
    '''
    # split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, stratify=y)
    
    # center and standardize x values
    x_scaler = StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    test_set = x_scaler.transform(test_set)
    
    return x_train, x_test, y_train, y_test, test_set


def export_csv(y, name):
    '''
    exports predicted labels into format to be submitted to Kaggle
    :param y: predicted labels
    :param name: name of file to be saved
    '''
    y = y.astype(int)
    df = pd.DataFrame({'Id':np.arange(1, 4321), 'Prediction':y})
    df.to_csv(name, sep=',', index=False)


# load data
print('Loading Data from S3...')
x = pickle.load(urllib.request.urlopen('https://s3.amazonaws.com/stat558drjordankaggle/train_features'))
y = np.array(pickle.load(urllib.request.urlopen('https://s3.amazonaws.com/stat558drjordankaggle/train_labels')))
y = y.astype(float)

# center and standardize all data to find optimal parameters
x_scaler = StandardScaler().fit(x)
x_train = x_scaler.transform(x)
y_train = y

# multinomial logistic regression cross-validation to find optimal regularization parameter
print('\nFinding Optimal Regularization Parameter for Multinomial Logistic...')
mult_logit = LogisticRegressionCV(fit_intercept=False, dual=False, multi_class='multinomial').fit(x_train, y_train)
logistic_opt_C = mult_logit.C_[0]
print('C = ', logistic_opt_C)

# multi-class (crammer-singer) linear SVC cross-validation to find optimal regularization parameter
print('Finding Optimal Regularization Parameter for LinearSVC (Crammer-Singer)...')
mult_linSVC_estimator = LinearSVC(multi_class='crammer_singer', fit_intercept=False)
parameters = {'C':[10**i for i in range(-4, -1)]}
mult_linSVC = GridSearchCV(mult_linSVC_estimator, parameters).fit(x_train, y_train)
linSVC_opt_C = mult_linSVC.best_params_['C']
print('C =', linSVC_opt_C)

# polynomial (order 2) kernel SVC cross-validation to find optimal regularization parameter
print('Finding Optimal Regularization Parameter for Polynomial (Order 2) Kernel SVC...')
poly2SVC_estimator = SVC(kernel='poly', degree=2)
parameters = {'C':[10**i for i in range(-1, 3)]}
poly2SVC = GridSearchCV(poly2SVC_estimator, parameters).fit(x_train, y_train)
poly2SVC_opt_C = poly2SVC.best_params_['C']
print('C =', poly2SVC_opt_C)

# initialize variables for iterating over models
test_all = np.arange(1, 4321)
num_iter = 10

for i in range(0, num_iter):
    
    # reload test data
    print('\nIteration:', (i + 1), 'of', num_iter)
    test_set = pickle.load(urllib.request.urlopen('https://s3.amazonaws.com/stat558drjordankaggle/test_features'))

    # create a new split of data to train model
    x_train, x_test, y_train, y_test, test_set = split_data_equal(x, y, test_set, train_size=0.75)

    # fit models with new train data
    print('Training Multinomial Logistic...')
    logistic = LogisticRegression(C=logistic_opt_C,
                                  fit_intercept=False,
                                  dual=False, solver='lbfgs',
                                  multi_class='multinomial').fit(x_train, y_train)
    
    print('Training LinearSVC (Crammer-Singer)...')
    linSVC = LinearSVC(C=linSVC_opt_C,
                       multi_class='crammer_singer',
                       fit_intercept=False).fit(x_train, y_train)
    
    print('Training Polynomial (Order 2) Kernel SVC...')
    polynomial2 = SVC(C=poly2SVC_opt_C, kernel='poly', degree=2).fit(x_train, y_train)
    
    # create predicted labels
    pred_logistic = logistic.predict(test_set)
    pred_linSVC = linSVC.predict(test_set)
    pred_poly2 = polynomial2.predict(test_set)
    
    # append all predicted labels
    test_all = np.vstack((test_all, pred_logistic, pred_linSVC, pred_poly2))

# aggregate predictions
df = pd.DataFrame(test_all[1:(num_iter*3 + 1)].T)
print('\nAll predictions:')
print(df)

# majority vote and export of file
final_prediction = np.ravel(stats.mode(test_all[1:(num_iter*3 + 1)].T, axis=1)[0])
df = pd.DataFrame(final_prediction)
print('\nFinal prediction:')
print(df)

# export to csv file for upload
export_csv(final_prediction, 'Kaggle_Attempt.csv')
print('\nCSV File Exported...Finished!')
