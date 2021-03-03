import os
import sys
import glob
from math import *
import csv
import random
import time
import datetime
import psutil
import copy

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *

dirKep = '49Ceti_model-Kep_csv/'
dirIRE = '49Ceti_model-IRE_csv/'
dirObs = '49Ceti_obs-regrid_csv/'


nowStr = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

dirLog = 'gscv_49Ceti/'
foutLogName = 'gscvResult_' + nowStr + '.log'
foutVarName = 'gscvResult_' + nowStr + '.variables.log'
foutScoreName = 'gscvResult_' + nowStr + '.epochScore.log'

f_saveLog = True
f_printLog = True

max_nfiles = 10000
max_iter = 1000
epochs = 100
max_batchsize = 512
n_splits = 5

test_size = 0.2
random_state = 0

n_jobs = -1
early_stopping = True

optimizers = [
                {
                'loss': ['log'],        #'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
                'alpha': [0.01],        # 0.0001, 0.001, 0.01, 0.1,
                'penalty': ['none']     # 'l2', 'l1', 'elasticnet', 'none'
                }
                ]


scoring = ["f1_macro"]

pf_shuffle = True


class myCLF(SGDClassifier):
    clfName = 'SGDClassifier'

clf = myCLF(max_iter = max_iter, n_jobs = n_jobs)

pipe = make_pipeline(StandardScaler(), clf)

###################################################

##########
# Header #
##########

labelKep = 0
labelIRE = 1
labelList = [labelKep, labelIRE]
labelName = ['Kep', 'IRE']
labelObs = 2

keyFileName = 'FileName'
keyData = 'Intensity'
keyLabel = 'ModelLabel'


timeStart = time.time()

try:
    os.mkdir(dirLog)
except:
    pass

foutLog = open(dirLog + foutLogName, 'w')
foutVar = open(dirLog + foutVarName, 'w')


def saveLog(comment = '', fout = foutLog, f_pLog = f_printLog, f_sLog = f_saveLog):
    if f_pLog: print(comment)
    if f_sLog: fout.write(comment + '\n')

def checkMemory():
    mem = psutil.virtual_memory()
    saveLog('\tMemory usage: {} %'.format(mem.percent))

def checkTime(time0 = timeStart):
    saveLog('\tDuration time: {} s'.format(time.time() - time0))
    
def checkStatus(commentPre = '', commentSuf = ''):
    saveLog('%s' % commentPre)
    checkMemory()
    checkTime()
    saveLog('%s' % commentSuf)


saveLog('\n----------Parameters----------')
saveLog('\nclasifier = {}'.format(clf.clfName))
saveLog('max_iter = {}'.format(max_iter))
saveLog('n_jobs = {}'.format(n_jobs))
saveLog('max_nfiles = {}'.format(max_nfiles))
saveLog('epochs = {}'.format(epochs))
saveLog('max_batchsize = {}'.format(max_batchsize))
saveLog('test_size = {}'.format(test_size))
saveLog('random_state = {}'.format(random_state))
saveLog('pf_shuffle = {}'.format(pf_shuffle))
saveLog('n_splits = {}'.format(n_splits))
saveLog('optimizers: ')
saveLog('\t{}'.format(optimizers))
saveLog('\nscoring: ')
saveLog('\t{}'.format(scoring))
saveLog('\n------------------------------')

checkStatus(commentPre = '\n\nExecuting: {}\n'.format(os.path.basename(__file__)))

#################
# Input Dataset #
#################

def makePDdataFrame(fileList, labelList):
    dataLoadList = []
    for filename in fileList:
        dataLoad = pd.read_csv(filename, usecols = [keyData], dtype = float)
        dataLoadList.append(dataLoad.values.T[0])
    return pd.DataFrame.from_dict({keyFileName: fileList, keyData: dataLoadList, keyLabel:labelList}, orient = 'columns')



fileListKep = glob.glob('%s/*.csv' % dirKep)
fileListIRE = glob.glob('%s/*.csv' % dirIRE)
fileListObs = glob.glob('%s/*.csv' % dirObs)

fileListKep = fileListKep[:min(len(fileListKep), max_nfiles)]
fileListIRE = fileListIRE[:min(len(fileListIRE), max_nfiles)]

saveLog('CSV Files to be load:\n')
saveLog('\t#Kepler model = %d' % len(fileListKep))
saveLog('\t#IRE    model = %d' % len(fileListIRE))
saveLog('\t#Observation  = %d' % len(fileListObs))

dataPDKep = makePDdataFrame(fileListKep, [labelKep for i in range(len(fileListKep))])
dataPDIRE = makePDdataFrame(fileListIRE, [labelIRE for i in range(len(fileListIRE))])
dataPDObs = makePDdataFrame(fileListObs, [labelObs for i in range(len(fileListObs))])


##################
# Split Datasets #
##################

checkStatus(commentSuf = '\nSplitting Models:')

X_trainKep, X_testKep, y_trainKep, y_testKep = train_test_split(dataPDKep[[keyFileName, keyData]], dataPDKep[keyLabel], test_size = test_size, random_state = random_state)
X_trainIRE, X_testIRE, y_trainIRE, y_testIRE = train_test_split(dataPDIRE[[keyFileName, keyData]], dataPDIRE[keyLabel], test_size = test_size, random_state = random_state)

checkStatus(commentSuf = '\nReshaping Datasets:')

X_train = np.array([item for item in pd.concat([X_trainKep[keyData], X_trainIRE[keyData]])])
y_train = pd.concat([y_trainKep, y_trainIRE]).values
X_test = np.array([item for item in pd.concat([X_testKep[keyData], X_testIRE[keyData]])])
y_test = pd.concat([y_testKep, y_testIRE]).values

fileName_obs = np.array(dataPDObs[keyFileName])
X_obs = np.array([item for item in dataPDObs[keyData]])
y_obs = np.array(dataPDObs[keyLabel])



###############
# Partial Fit #
###############


cv = StratifiedKFold(n_splits = n_splits)

param_grid = dict(optimizer = optimizers, epochs = epochs, early_stopping = early_stopping) #, init = init) #batch_size = batch_size




def partialFit(clf = clf, foutScoreName = foutScoreName):
    foutScore = open(dirLog + foutScoreName, 'w')
    n_minibatch = ceil(len(X_train) / max_batchsize)
    checkStatus(commentSuf = '\nPartial Fit:\n\t\t#epoch = {}\n\t\t#train data    = {}\n\t\tmax batch_size = {}\n\t\t-> #mini batch = {}\n\t\t   batch_size = {}'.format(epochs, len(X_train), max_batchsize, n_minibatch, len(X_train) / n_minibatch))
    foutScore.write('#Estimator: {}\n#\t\t{}\n'.format(clf, clf.__dict__))
    foutScore.write('#Epoch\tScore\n')
    for iepoch in range(epochs):
        pf = StratifiedKFold(n_splits = n_minibatch, shuffle = pf_shuffle, random_state = iepoch)
        saveLog('\n\tEpoch ({} / {}):'.format(iepoch + 1, epochs))
        validIDlist = []
        currentAccuracy = 0.
        for ibatch, (dummy, batch_index) in enumerate(pf.split(X_train, y_train)):
            if ibatch == 0:
                validIDlist = batch_index
                saveLog('\t\tID list for validation: {}'.format(validIDlist))
                continue
            clf.partial_fit(X_train[batch_index], y_train[batch_index], classes = labelList)
            y_pred = clf.predict(X_train[validIDlist])
            currentAccuracy = accuracy_score(y_train[validIDlist], y_pred)
            saveLog('\t\tmini batch ({} / {}) -> accuracy score = {}'.format(ibatch, n_minibatch - 1, currentAccuracy))
        #best_estimator = clf
        saveLog('\n\t\t-> The last score of this epoch ({} / {}): {}'.format(iepoch + 1, epochs, currentAccuracy))
        foutScore.write('{}\t{}\n'.format(iepoch + 1, currentAccuracy))
    foutScore.write('\n')
    foutScore.close()

checkStatus(commentSuf = '\nGrid Search:')
bestscore = -1.
bestParDict = {}
for optDict in optimizers:
    keyList = optDict.keys()
    nkey = len(keyList)
    nvalList = []
    nvalProdList = []
    for lis in optDict.values():
        nvalList.append(len(lis))
    for ikey in range(nkey):
        nvalProdList.append(np.prod(nvalList[ikey:]))
    nvalProdList.append(1)
    for idxSerial in range(nvalProdList[0]):
        saveLog('\n-----Grid Search ({} / {})-----\n'.format(idxSerial + 1, nvalProdList[0]))
        idxList = []
        for ikey in range(nkey):
            idxList.append(int(idxSerial / nvalProdList[ikey + 1]) % nvalList[ikey])
        parDict = {}
        for ikey, key in enumerate(keyList):
            parDict[key] = optDict[key][idxList[ikey]]
            clf.__dict__[key] = parDict[key]
        saveLog('\nclassifier: {}\n\t{}'.format(clf, clf.__dict__))
        partialFit(clf, foutScoreName = foutScoreName[:foutScoreName.rfind('.') + 1] + str(idxSerial + 1) + foutScoreName[foutScoreName.rfind('.'):])
        y_pred = clf.predict(X_test)
        lastscore = accuracy_score(y_test, y_pred)
        saveLog('\nLast score for this estimator = {}'.format(lastscore))
        if bestscore < lastscore:
            bestscore = lastscore
            bestParDict = parDict.deepcopy()
best_estimator = clf
for key in bestParDict:
    bestestimator.__dict__[key] = bestParDict[key]
saveLog('\n\n----------\nBest Estimator: {} \n\tscore = {}\n\tparameters = {}\n\n\tclassifier = {}\n\t\t{}'.format(best_estimator.clfName, bestscore, bestParDict, best_estimator, best_estimator.__dict__))



########
# Test #
########


def saveLog_confusionMatrix(y_true, y_pred):
    saveLog('-----\nConfusion Matrix:')
    ans = confusion_matrix(y_true, y_pred)
    saveLog('\tAnswer | Prediction')
    saveLog('\t       | ' + '\t'.join(str(item) for item in labelList))
    for i, row in enumerate(ans):
        strList = [str(labelList[i]) + '|']
        strList.extend(str(row))
        saveLog('\t {:>5} | '.format(str(labelList[i])) + '\t'.join(str(item) for item in row))
    saveLog('')

def saveLog_classificationReport(y_true, y_pred):
    saveLog('-----\nClassification Report:')
    saveLog(classification_report(y_true, y_pred))
    
    
checkStatus(commentSuf = '\nTesting:')
y_pred = best_estimator.predict(X_test)

checkStatus(commentSuf = '\nReport: ')

saveLog_confusionMatrix(y_test, y_pred)
saveLog_classificationReport(y_test, y_pred)



##############
# Prediction #
##############


checkStatus(commentSuf = '\nPredicting:')
y_obspred = best_estimator.predict(X_obs)

checkStatus(commentSuf = '\nResults: ')
saveLog('\tPrediction| Observation File Name')
for i in range(len(y_obspred)):
    saveLog('\t     {}    | {}'.format(y_obspred[i], fileName_obs[i]))
saveLog('\n\n\t(*) Label list: ')
for i in range(len(labelList)):
    saveLog('\t\t{} -- {}'.format(labelList[i], labelName[i]))


#########
# Close #
#########

foutVar.write('\nResult Files\n')
foutVar.write('\tLog  : {}\n'.format(foutLogName))
#foutVar.write('\tScore: {}\n'.format(foutCsvName))
foutVar.write('\n\n\n----------Variables----------\n')
foutVar.write('{}'.format(locals().items()))




checkStatus(commentSuf = '\n\nFinished: {}\n\n'.format(os.path.basename(__file__)))

print('\nSaved: ')
print('\tLog                        : %s' % foutLogName)
#print('\tGridSearch CrossVal results: %s\n\007\n' % foutCsvName)


foutLog.close()
#foutCsv.close()
foutVar.close()

