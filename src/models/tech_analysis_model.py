import dataExtractor
import pandas as pd
import sklearn
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

##Data_df = (dataExtractor.createCompleteDataset("BTC-USD", "2007-12-31", "2017-12-31", 10))
##coin_data = (dataExtractor.getCoinData("BTC-USD", "2007-12-31", "2017-12-31"))
##close_price_df = pd.DataFrame(coin_data['Adj. Close'])
##
##writer = pd.ExcelWriter("coinDataFrame.xlsx", engine = 'xlsxwriter')

##Data_df.to_excel(writer, sheet_name = 'DataFrame')

##writer.save()

XLdf = pd.ExcelFile("coinDataFrame.xlsx")

Data_df = XLdf.parse("DataFrame")

X = np.array(Data_df.drop(['Up or Down?'], 1))
Y = np.array(Data_df['Up or Down?'])

X = preprocessing.scale(X)

accTotal = 0
aucTotal = 0
precisionTotal = 0
recallTotal = 0
f1Total = 0

i = 0
while i < 100:
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = KNeighborsClassifier()

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, Y_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)

    accTotal = accTotal + accuracy
    aucTotal = aucTotal + auc
    precisionTotal = precisionTotal + precision
    recallTotal = recallTotal + recall
    f1Total = f1Total + f1

    i = i + 1
avgAccuracy = accTotal/100
avgAUC = aucTotal/100
avgPrecision = precisionTotal/100
avgRecall = recallTotal/100
avgF1 = f1Total/100

print("Average AUC: " + str(avgAUC))
print("Average Accuracy: " + str(avgAccuracy))
print("Average Precision: " + str(avgPrecision))
print("Average Recall: " + str(avgRecall))
print("Average F1 Score: " + str(avgF1))
