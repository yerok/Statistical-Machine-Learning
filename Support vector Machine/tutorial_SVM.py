
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import validation_curve
from sklearn.metrics import zero_one_loss

import matplotlib.pyplot as plt

from math import log,sqrt
import numpy as np
import scipy.sparse


def training():
    # pre computed kernel Matrix 
    # Read data in LIBSVM format
    # In this module, scipy sparse CSR matrices are used for X and numpy arrays are used for y.
    trnKernelMatrix, trnLabels = load_svmlight_file("./dns_data_kernel/trn_kernel_mat.svmlight")
    valKernelMatrix, valLabels = load_svmlight_file("./dns_data_kernel/val_kernel_mat.svmlight")

    print(trnKernelMatrix)
    C = [0.01,0.1,1,10,100]
    table = []

    # conversion because fit() function doesn't work with CSR matrices and precompputed kernel
    trnKernelMatrix = trnKernelMatrix.toarray()
    valKernelMatrix = valKernelMatrix.toarray()

    #Training and Prediction for each value of C
    for index, c in enumerate(C):
        clf = svm.SVC(kernel='precomputed',C=c)
        fit = clf.fit(trnKernelMatrix,trnLabels)

        #Prediction
        trnPredict = clf.predict(trnKernelMatrix)
        valPredict = clf.predict(valKernelMatrix)

        #Empirical risk with 0/1 loss  (fraction of misclassifications)
        trainingError = zero_one_loss(trnLabels,trnPredict)
        validationError = zero_one_loss(valLabels, valPredict)

        #mean accuracy
        trainingScore = fit.score(trnKernelMatrix,trnLabels)
        validationScore =  fit.score(valKernelMatrix,valLabels)
        supportVectors = clf.support_

        svmInfo = [c, trainingError, validationError, len(supportVectors)]
        table.append(svmInfo)

    return table

def testing():
    trnKernelMatrix, trnLabels = load_svmlight_file("./dns_data_kernel/trn_kernel_mat.svmlight")
    testKernelMatrix, testLabels = load_svmlight_file("./dns_data_kernel/tst_kernel_mat.svmlight")
    trnKernelMatrix = trnKernelMatrix.toarray()
    testKernelMatrix = testKernelMatrix.toarray()


    clf = svm.SVC(kernel='precomputed',C=1)
    clf.fit(trnKernelMatrix,trnLabels)

    testPredict = clf.predict(testKernelMatrix)

    #Empirical risk with 0/1 loss  (fraction of misclassifications)
    testError = zero_one_loss(testLabels, testPredict)

    print("Error with testing data : ", testError)



def output(table):

    for i in range(0,len(table)):
        print("C = ", table[i][0])
        print("Training error  = ", table[i][1])
        print("Validation Error  ", table[i][2])
        print("Number of support vectors = ", table[i][3])


#We can see that the lowest validation error (0.068) is obtained with C = 1
def plotErrorWrtC(table):

    C = []
    trainingError = []
    validationError = []

    for i in range(0,len(table)):
        C.append(table[i][0])
        trainingError.append(table[i][1])
        validationError.append(table[i][2])

    plt.plot(C,trainingError,label="Training error")
    plt.plot(C,validationError,label="validation error")

    plt.xscale('log')
    plt.xlabel('Value of C')
    plt.ylabel('Fraction of misclassification')
    plt.legend()
    plt.title('Errors as a function of C')
    plt.show()



def getEpsilon(a,b,gamma,numberOfExamples):

    l = numberOfExamples

    # e = abs(b-a)*sqrt((log(2)-log(1-gamma)/2*l)
    e = abs(a-b)*sqrt((log(2)-log(1-gamma))/(2*l))
    return e 


def main():
    table = training()

    print("\n --- Training with differents values of C --- \n")    
    output(table)

    # We choose C = 1
    print("\n --- Predicting on test data ---")
    testing()

    print("\n --- computing the minimal value of epsilon --- ")
    print("Minimal value of epsilon : ",getEpsilon(1,-1,0.99,2000))

    plotErrorWrtC(table)

if __name__ == "__main__": main()

