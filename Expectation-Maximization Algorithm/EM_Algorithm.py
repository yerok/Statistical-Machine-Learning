import numpy as np
from sklearn.metrics import zero_one_loss
from PIL import Image
from sklearn.cluster import KMeans
import glob
from sklearn import mixture
from scipy.stats import multivariate_normal
from os import listdir
from os.path import isfile, join


def loadingImages():

    imgs = []
    imgsSeg = []

    for filename in glob.glob('./shape_em/data/*.png'): 
        if "seg" in filename:
            # GrayScale
            im=Image.open(filename).convert('L')
            arr = np.array(im, dtype=np.float32) / 255.0
            imgsSeg.append(arr)
        elif "hand" in filename :
            # RGB
            im=Image.open(filename).convert("RGB")
            arr = np.array(im, dtype=np.float32) / 255.0
            imgs.append(arr)
        elif "init" in filename: 
            im=Image.open(filename).convert('L')
            modelInit = np.array(im, dtype=np.float32) / 255.0

    imagesMat = np.array(imgs, dtype=np.float32)
    imagesSegMat = np.array(imgsSeg, dtype=np.float32)

    return [imagesMat,imagesSegMat, modelInit]


def getMeans(image):

    # We calculate means of RGB of the first column
    backgroundMean = np.mean(image[:,0],axis=0)

    width = image.shape[1]
    height = image.shape[0]

    #We create a little square in the center of the image for the foreground (simpler than a circle)
    squareTop = int((height/2)+10)
    squareBottom = int((height/2)-10)
    squareLeft = int((width/2)-10)
    squareRight = int((width/2)+10)

    foregroundMean = np.mean(image[squareBottom:squareTop,squareLeft:squareRight],axis=(0,1))


    return backgroundMean, foregroundMean

def kmeans(initMeans,image):

    w,h,value = image.shape
    image2D = image.reshape((w*h,value))

    kmeans = KMeans(n_clusters=2,init=np.array(initMeans),n_init=1)

    fit = kmeans.fit(image2D)

    pred = kmeans.predict(image2D)

    pred = pred.reshape(w,h)

    return pred

def evaluateKmeans(images,imagesSeg):

    predictions = []
    error = 0

    for image in images:
        initMeans = getMeans(image)
        pred = kmeans(initMeans,image)
        predictions.append(pred)

    predictions = np.array(predictions)
    # error = zero_one_loss(imagesSeg,predictions)    

    n,w,h = predictions.shape
    pred = predictions.reshape(n*w*h)
    true = imagesSeg.reshape(n*w*h)
    true = true.astype(int)

    # error = np.mean(true != pred)
    error = zero_one_loss(true,pred)    

    return error

def gaussianMixture(initMeans,image):

    w,h,value = image.shape
    image2D = image.reshape((w*h,value))

    gaussianMixture = mixture.GaussianMixture(n_components=2,means_init=np.array(initMeans),n_init=1,covariance_type='full')

    fit = gaussianMixture.fit(image2D)

    pred = gaussianMixture.predict(image2D)

    pred = pred.reshape(w,h)

    return pred

def evaluateGaussianMixture(images,imagesSeg):

    predictions = []
    error = 0
    for image in images:
        initMeans = getMeans(image)
        pred = gaussianMixture(initMeans,image)
        predictions.append(pred)

    predictions = np.array(predictions)

    n,w,h = predictions.shape
    pred = predictions.reshape(n*w*h)
    true = imagesSeg.reshape(n*w*h)
    true = true.astype(int)

    # error = np.mean(pred != true)
    error = zero_one_loss(true,pred)    

    return error


# Not complete
def EM(images,imagesSeg,shapeModel):

    numberOfImages,h,w,value = images.shape

    a = np.zeros((numberOfImages,h,w))
    newA = np.zeros((numberOfImages,h,w))
    u = np.array(shapeModel)

    # init of a 
    a[:] = u

    meanTheta0 = np.zeros((numberOfImages,value))
    meanTheta1 = np.zeros((numberOfImages,value))
    covTheta0 = np.zeros((numberOfImages,value,value))
    covTheta1 = np.zeros((numberOfImages,value,value))

    # We create a numpy array which store objects (multivariate_normal_frozen)
    probaDensityFunction = np.empty(shape=(numberOfImages,), dtype=object)

    testImage = images[0]
    h,w,value = testImage.shape 

    # init of thetas 
    aImage = a[0]
    # print(aImage.shape)
    posPixels = testImage[aImage >= 0.5]
    negPixels = testImage[aImage < 0.5]

    # print(posPixels.shape)
    meanTheta0[:] = np.mean(negPixels[0],axis=(0))   
    meanTheta1[:] = np.mean(posPixels[0],axis=(0))

    # rowvar set to false to precise that columns are variables
    covTheta0[:] = np.cov(negPixels,rowvar=False)
    covTheta1[:] = np.cov(posPixels,rowvar=False)

    # probability density function of normally distributed random variable (gaussian)
    probaDensityFunction[:] = multivariate_normal(meanTheta1[0], covTheta1[0])
    maxIter = 100


    for iteration in range(maxIter):


        # E-step        
        # for i in range(len(images)):    
        #     newA[i] = probaDensityFunction[:].pdf()

        # M-Step
        u = np.log(newA/(1 - newA))

        for i in range(len(images)):
            image = images[i]
            aImage = newA[i]
            print(aImage.shape)

            posPixels = image[aImage >= 0.5]
            negPixels = image[aImage < 0.5]

            meanTheta0[i] = np.mean(negPixels[0],axis=(0))  
            meanTheta1[i] = np.mean(posPixels[0],axis=(0))

            # rowvar set to false to precise that columns are variables
            covTheta0[i] = np.cov(negPixels,rowvar=False)
            covTheta1[i] = np.cov(posPixels,rowvar=False)

            # probability density function of multivariate normally distributed random variable (gaussian)
            probaDensityFunction[i] = multivariate_normal(meanTheta1[i], covTheta1[i])

        # convergence check
        if (np.mean(np.abs(a-newA)) < 0.05):
            break
        a = newA
        first = False

    a[a >= 0.5] = 1
    a[a < 0.5] = 0

    n,w,h = a.shape

    true = imagesSeg.reshape(n*w*h)
    true = true.astype(int)
    a = a.reshape(n*w*h)
    a = a.astype(int)

    error = zero_one_loss(true,a)    
    print(error)



def main():
    images = loadingImages()
    imagesMat = images[0]
    imagesSegMat = images[1]
    shapeModel = images[2]

    errorKmeans = evaluateKmeans(imagesMat,imagesSegMat)
    print("Error with K-means : ", errorKmeans)
    errorGaussianMixture = evaluateGaussianMixture(imagesMat,imagesSegMat)
    print("Error with Gaussian Mixture : ",errorGaussianMixture)

    # EM(imagesMat,imagesSegMat,shapeModel)

if __name__ == "__main__":
    main()