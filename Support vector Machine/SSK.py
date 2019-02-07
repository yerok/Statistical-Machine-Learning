import numpy as np
from math import sqrt


def sskArray(str1,str2,q,decay):

	str1Array = list(str1)
	str2Array = list(str2)

	lenStr1 =  len(str1Array)
	lenStr2 =  len(str2Array)

	kPrime = np.zeros((q,lenStr1,lenStr2),dtype=np.float)

	kPrime[0,:,:] = 1

	for n in range(1,q):
		for i in range(n,lenStr1):
			phi = 0
			for j in range(n,lenStr2):
				if str1Array[i-1]==str2Array[j-1]:
					phi = decay * (phi + decay*kPrime[n-1,i-1,j-1])
				else:
					phi *= decay
				kPrime[n,i,j] = phi + decay * kPrime[n,i-1,j]


	k = 0

	for n in range(q):
		for i in range(n,lenStr1):
			for j in range(n,lenStr2):
				k += decay*decay*kPrime[n,i,j]

	return k 


def ssk(str1,str2,q,decay,normalization):


	lenStr1 =  len(str1)
	lenStr2 =  len(str2)

	gramMatrix = np.zeros((lenStr1,lenStr2),dtype = np.float)


	# avoid recalculation if it's the same list (symetric matrix)
	if str1 == str2:
		for i in range(lenStr1):
			for j in range(i,lenStr2):
				gramMatrix[i,j] = gramMatrix[j,i] = sskArray(str1[i],str2[j],q,decay)

		matStr1 = matStr2 = gramMatrix.diagonal().reshape( (lenStr1, 1) )		
	else :
		for i in range(lenStr1):
			for j in range(lenStr2):
				gramMatrix[i,j] = sskArray(str1[i],str2[j],q,decay)

		matStr1 = np.zeros((lenStr1,1))
		matStr2 = np.zeros((lenStr2,1))

	if normalization:
		for i in range(lenStr1):
			matStr1[i] == sskArray(str1[i],str1[i],q,decay)
		for j in range(lenStr2):
			matStr2[j] == sskArray(str2[j],str2[j],q,decay)	

		# Normalization 
		return np.divide(gramMatrix,np.sqrt(matStr2.T * matStr1))
	else:
		return gramMatrix

def main():

	str1 = ["google.com","facebook.com","atqgkfauhuaufm.com","vopydum.com"]
	str2 = ["google.com","facebook.com","atqgkfauhuaufm.com","vopydum.com"]
	print("\nGram matrix of SSK witout normalization\n")
	print(ssk(str1,str2,3,0.4,False))
	print("\nGram matrix of SSK with normalization\n")
	print(ssk(str1,str2,3,0.4,True))

if __name__ == "__main__": main()

