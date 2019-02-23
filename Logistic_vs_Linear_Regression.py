import csv

'''Handle data'''
def load_csv(filename):
  lines = csv.reader(open(filename))
  dataset = list(lines)
  
  for i in range(1, len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  
  header = list(dataset)[0]
  dataset = list(dataset)[1:]

  return dataset, header

filename= 'pima-indians-diabetes.csv'
dataset, header= load_csv('diabetes.csv')
print('loaded {} with {} rows'.format(filename,len(dataset)))

'''split data'''
import random
def split_data(dataset, split_ratio):
  trainSize = round(len(dataset)*split_ratio)
  trainSet = []
  testSet = []
  random.shuffle(dataset) 
  trainSet = dataset[:trainSize]
  testSet= dataset[trainSize:]
  return trainSet, testSet

#trainSet, testSet = split_data(dataset, 0.8)

#print('split {} rows into train with {} and test with {}'.format
#(len(dataset),len(trainSet),len(testSet)))


import math

def predict(row, coefficients):
  #print('row=',row,'coefficient=',coefficients)
  yhat = coefficients[0]
  for i in range(len(row)-1):
    yhat += coefficients[i+1]*row[i]
  # the if statement is to avoid overflow
  if yhat >= 0:
    prob = 1/(1 + math.exp(-yhat))
  elif yhat < 0:
    prob = 1 - 1/(math.exp(yhat)+ 1)
  #print('yhat', yhat, 'prob', prob)
  return prob

data_test = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

coef = [-0.406605464, 0.852573316, -1.104746259]


#for row in data_test:
#  yhat = predict(row, coef)
#  print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))

'''stochastic gradient descent'''
# http://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/06_GenDiscr_LR_2-2-2015-ann.pdf

# Maximizing Conditional Log Likelihood: l(W) = sum(P(Y`l|X`l,W) for all <X`l, Y`l> in train set L
# After drivation, partial derivative of l(W) with respective of wi can be used for gradient ascent 

# 1. loop each epoch
# 2. loop each row in trainSet for each epoch
# 3. loop each coef for each row of trainset for each epoch
# also record error for each epoch
def coef_update_sgd(trainSet, l_rate, n_epoch):
  coef_len = len(trainSet[0]) # constant coef + 1, label - 1
  coef = [0 for _ in range(coef_len)]
  for epoch in range(n_epoch):
    #print(coef)
    error_sum = 0
    prob_sum = 0
    for row in trainSet:
      prob_hat = predict(row, coef)
      error = row[-1] - prob_hat
      error_sum += error**2
      if row[-1] == 1:
        prob_sum += math.log(prob_hat)
      else:
        prob_sum += math.log(1-prob_hat)
      #print('prob_hat', prob_hat, 'error', error)
      coef[0] = coef[0] + l_rate*error
      for i in range(1, coef_len):
        coef[i] = coef[i] + l_rate*error*row[i-1] 
    #print('epoch=%s, l_rate=%.2f, error_sum=%.3f, logMCLE=%.1f' % (epoch, l_rate, error_sum, prob_sum))
  return coef

#print(coef_update_sgd(data_test, 0.5, 100))
def coef_update_gABatch(trainSet, l_rate, n_epoch):
  coef_len = len(trainSet[0]) # constant coef + 1, label - 1
  coef = [0 for _ in range(coef_len)]
  for epoch in range(n_epoch):
    #print(coef)
    error_sum = 0
    prob_sum = 0
    # place hold to get batch gradient
    gradient = [0 for _ in range(coef_len)]
    for row in trainSet:
      prob_hat = predict(row, coef)
      error = row[-1] - prob_hat
      error_sum += error**2
      if row[-1] == 1:
        prob_sum += math.log(prob_hat)
      else:
        prob_sum += math.log(1-prob_hat)
      '''gradient for each w_i for all rows'''
      # assume x0 = 1 for all records
      gradient[0] += 1*error
      # calculating batch gradient at each w_i
      for i in range(1, coef_len):
        gradient[i] += row[i-1]*error
      #print('prob_hat', prob_hat, 'error', error)
    
    # update each w_i using l_rate*gradient at each w_i
    for i in range(coef_len):
      coef[i] = coef[i] + l_rate*gradient[i]
    
    print('epoch=%s, l_rate=%.2f, error_sum=%.3f, logMCLE=%.1f' % (epoch, l_rate, error_sum, prob_sum))
  return coef

''' apply logistic regression on the PIMA Diabete Dataset'''

'''rescale data'''
# if data is not rescaled, the yhat = b0 + b1*X becomes a big integer, causings the logit function gives 0 or 1 probablity 
# get all the minimum and maximum of each col of the dataset
def get_minmax(dataset):
  n_col = len(dataset[0])
  minmax = []
  for i in range(n_col):
    col_value = [row[i] for row in dataset]
    min_col = min(col_value)
    max_col = max(col_value)
    minmax.append([min_col, max_col])
  return minmax
  
print(get_minmax(dataset),'minmax')
minmax = get_minmax(dataset)


import copy
#scaling
def scale_dataset(dataset, minmax):
  # deep copy works for 2D data
  scaled_set = copy.deepcopy(list(dataset[:]))
  n_col = len(scaled_set[0]) - 1 
  n_row = len(scaled_set)
  for j in range(n_col):
    for i in range(n_row):
      scaled_set[i][j] = (scaled_set[i][j] - minmax[j][0])/(minmax[j][1] - minmax[j][0])
  return scaled_set

data_scaled = scale_dataset(dataset, minmax)


''' train logistic regression model using stochastic gradient discent
# use k-fold cross validation to estimate the performance of unseen data'''
# 1. split data set
# 2. for each fold, train (stochastic gd) model on the k-1 folds, and test on the kth fold. Get the average accuracy of all k rounds of training. Goal is to pick the coefficent that minize result of CV 


# split a dataset into k folds
from random import randrange
def cross_validation_split(dataset,n_folds):
  dataset_split = list()
  dataset_copy = list(dataset)
  fold_size = int(len(dataset)/n_folds)
  
  for i in range(n_folds):
    fold = list()
    while len(fold) < fold_size:
      index = randrange(len(dataset_copy))
      fold.append(dataset_copy.pop(index))
    dataset_split.append(fold)
  print ('record used',len(fold)*n_folds)
  return dataset_split

#test_folds = (list(cross_validation_split(trainSet,5)))
#print('len of a0',len(test_folds), len(sum(test_folds,[])))


# calc accuracy score
def accuracy_metric(y_true, y_pred):
  count = 0
  for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
      count += 1
  score = count/len(y_true)
  return score

# evaluate algorithm using cross validation

def evaluate_algorithm(trainSet, n_folds, pred_function, algorithm, *args):

# split data into k folds
  folds = cross_validation_split(trainSet, n_folds)
  scores = []
  for fold in folds:
    trainSet_cv = list(folds)
    trainSet_cv.remove(fold)
    trainSet_cv = sum(trainSet_cv, [])
    testSet_cv = copy.deepcopy(fold)
    coef = algorithm(trainSet_cv, *args)
    y_pred = pred_function(testSet_cv,coef)
    y_true = [row[-1] for row in testSet_cv]
    
    print(testSet_cv, y_pred)
    accuracy = accuracy_metric(y_true, y_pred)
    scores.append(accuracy)

  return scores

# fit model using logistic regression on the kth fold, and predict on the kth fold
  

#print(evaluate_algorithm(trainSet, logistic_train, 5))


def predict_all(testSet, coef):
  y_pred = []
  for row in testSet:
    pred = round(predict(row, coef))
    y_pred.append(pred)
  return y_pred

#y_pred = [round(row) for row in predict_all(testSet, coef)]

#y_true = [row[-1] for row in testSet]
#print(accuracy_metric(y_true, y_pred))

#minmax = get_minmax(trainSet)
#trainScaled = scale_dataset(trainSet, minmax)
minmax_whole = get_minmax(dataset)
dataScaled = scale_dataset(dataset, minmax_whole)
#scores = evaluate_algorithm(dataScaled, 5, predict_all, coef_update_sgd, 0.1, 100)
#print(sum(scores)/5, 'scores_logistic')


'''now try linear regression'''
def predict_LinReg(row, coefficent):
  yhat = coefficent[0]
  coef_len = len(coefficent)
  for i in range(1, coef_len):
    yhat += coefficent[i] * row[i -1]
  return yhat 

def coef_update_sgd_LinReg(trainSet, l_rate, n_epoch):
  coef_len = len(trainSet[0]) + 1 - 1
  coef = [0 for _ in range(coef_len)]
  for epoch in range(n_epoch):
    sum_error = 0
    for row in trainSet:
      y_hat = predict_LinReg(row, coef)
      error = row[-1] - y_hat
      #print(error, coef)
      sum_error += error**2
      coef[0] = coef[0] - l_rate * (-error)
      for i in range(1, coef_len):
        coef[i] = coef[i] - l_rate*(-error*row[i-1])
    #print('epoch=%d,l_rate=%.3f, sum_error=%.2f' % (epoch,l_rate,sum_error))
  return coef

def predict_all_LinReg(testSet, coef):
  y_pred = []
  for row in testSet:
    pred = round(predict_LinReg(row, coef))
    y_pred.append(pred)
  return y_pred


#trainScaled, testScaled = split_data(dataScaled, 0.8)

#coef_LinReg = coefficients_update_sgd_LinReg(trainScaled, 0.01, 40)

#y_train_pred = predict_all_LinReg(trainScaled,coef_LinReg)
#y_train_true = [row[-1] for row in trainScaled]

#for i in range(30):
#  print(y_train_pred[i],y_train_true[i])

#print(sum(y_train_pred)/len(y_train_pred))
#print(sum(y_train_true)/len(y_train_true))

#y_pred = predict_all_LinReg(testScaled,coef_LinReg)
#y_true = [row[-1] for row in testScaled]
#accuracy_train = accuracy_metric(y_train_true, y_train_pred)
#print(accuracy_train,'accuracy_train')

#accuracy_test = accuracy_metric(y_true, y_pred)
#print(accuracy_test,'accuracy_test')

#scores_LinReg = evaluate_algorithm(dataScaled, 5, predict_all_LinReg, coefficients_update_sgd_LinReg, 0.01, 40)
#print(sum(scores_LinReg)/5,'LinReg')

'tumor size data'
tm_header = ['tumor size','malignant']
tm_data = [
  [1, 0],
  [2, 0],
  [3, 0],
  [4, 1],
  [5, 1],
  [6, 1],
  [20,1]
]

td_minmax = get_minmax(tm_data)
td_scaled = scale_dataset(tm_data, td_minmax)


scores = evaluate_algorithm(td_scaled, 5, predict_all, coef_update_sgd, 10, 30)
print(sum(scores)/5, 'scores_logistic')

scores_LinReg = evaluate_algorithm(td_scaled, 5, predict_all_LinReg, coef_update_sgd_LinReg, 0.2, 30)
print(sum(scores_LinReg)/5,'LinReg')
