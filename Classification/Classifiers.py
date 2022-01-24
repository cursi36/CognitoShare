import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #SVC = SVM classifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder #creates array for each possible class e.e a = [1 0 0],b = [0 1 0], c = [0 0 1]
from sklearn.preprocessing import LabelEncoder #assigns o-n-1 to each class e.g a = 0, b= 1, c= 2
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


def getData():
    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    # Read dataset to pandas dataframe
    irisdata = pd.read_csv("iris.data", names=colnames)
    print(irisdata.head())

    #use label Encoder to assign values 0,1,2 to iris classes
    le = LabelEncoder()
    le.fit(irisdata['Class'])

    print("label encoded classes:", le.classes_)
    class_enc = le.transform(irisdata['Class'])
    # print("Data transformed encoded:" , class_enc)

    #set the class column to be the encoded values
    irisdata['Class'] = class_enc

    X = irisdata.drop('Class', axis=1)  # x is all columns except class column
    y = irisdata['Class']
    print(X.head())
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train,y_train,X_test,y_test

def EvaluateResults(y_pred,y_test):
    print("*****CONFUSION MATRIX****")
    print(confusion_matrix(y_test, y_pred))
    print("*****OVERALL RES****")
    print(classification_report(y_test, y_pred))

def LogisticRegr(X_train,y_train,X_test):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    # print(clf.predict(X_train[:2, :]))
    # print(clf.predict_proba(X_train[:2, :]))
    # print(clf.score(X_train, y_train))

    y_pred = clf.predict(X_test)

    return y_pred


def SVMClassifier(X_train,y_train,X_test):
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    return y_pred

def DecisionTree(X_train,y_train,X_test):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)


    fig = plt.figure(figsize=(5, 5))
    _ = tree.plot_tree(clf,
                       feature_names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'],
                       class_names=["0","1","2"],
                       filled=True)
    plt.show()
    return y_pred

def NaiveBayes(X_train,y_train,X_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    return y_pred

def KNN(X_train,y_train,X_test):
    model = KNeighborsClassifier(n_neighbors=10, weights='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred

if __name__=="__main__":

    X_train,y_train,X_test,y_test = getData()

    print("******Logistic regression")
    y_logRegr = LogisticRegr(X_train,y_train,X_test)
    EvaluateResults(y_logRegr, y_test)

    print("******Naive Bayes")
    y_NaiveBayes = NaiveBayes(X_train, y_train, X_test)
    EvaluateResults(y_NaiveBayes, y_test)

    print("******KNN")
    y_KNN = KNN(X_train, y_train, X_test)
    EvaluateResults(y_KNN, y_test)

    print("******SVM")
    y_SVM = SVMClassifier(X_train, y_train, X_test)
    EvaluateResults(y_SVM , y_test)

    print("******Decision Tree")
    y_DT = DecisionTree(X_train, y_train, X_test)
    EvaluateResults(y_DT, y_test)





