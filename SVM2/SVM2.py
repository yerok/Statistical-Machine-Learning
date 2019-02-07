import scipy.io
import numpy as np
import time
import math
import random
from os import listdir
from os.path import isfile, join
import os, shutil

def loadFiles(path):

	# we get the name of the files (in fact useless for mat format)
 	# files = [f for f in listdir(path) if isfile(join(path, f))]
	# filePath = path  + files[i]

	mat = scipy.io.loadmat(path)
	trainData = mat['TrnData']
	testData = mat['TstData']

	return trainData,testData

def sequenceError(pred,true):

	error = np.mean(pred != true)
	return error

def characterError(pred,true):

	error = np.mean(pred != true, axis=(0,1))

	return error

# linear multi-class classifier
def perceptron1(data):

	imgs = data['img']
	features = data['X']
	characters = data['Y']


	learningRate = 0.1
	
	#Number of characters
	length = int(int(imgs.shape[1])/8)

	#Number of features
	nbFeatures = features.shape[0]

	#alphabet 
	nbCharacters = 26

	# init weights
	weights = np.zeros(nbFeatures)
	weights[:] = random.random()
    
    #init of biases
	bias = np.ones(length)
	biasWeights = np.zeros(length)
	biasWeights[:] = random.random()

	# we transpose the matrix otherwise it seems to not be working in the next loop
	features = features.T
	print(features.shape)





	# for i in range(100):
	# while True
	# 	for i in range(length):
	# 		feature = features[:][i]
	# 		res = np.dot(feature,weights) + np.dot(bias,biasWeights)
	# 		error = 
	#         for index, value in enumerate(input_vector):
	#             weights[index] += learningRate * error * value

	# 		print(res)

    # while True:
    #     error_count = 0
    #     for input_vector, desired_output in training_set:
    #         result = dot_product(input_vector, weights) > threshold
    #         error = desired_output - result
    #         if error != 0:
    #             error_count += 1
    #             for index, value in enumerate(input_vector):
    #                 weights[index] += learning_rate * error * value
    #     if error_count == 0:
    #         break


def main():
	trainData,testData = loadFiles("./ocr_names.mat")


	# trainImgs = trainData['img']
	# trainFeatures = trainData['X']
	# trainCharacters = trainData['Y']

	perceptron1(trainData[0][0])

	# testImgs = testData['img']
	# testFeatures = testData['X']
	# testCharacters = testData['Y']

if __name__ == "__main__":
    main()