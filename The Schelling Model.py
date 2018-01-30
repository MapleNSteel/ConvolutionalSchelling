import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2

numDem=500
numRep=500

numColumns=50
numRows=50

numGrid=numColumns*numRows

windowSize=3

kernel=np.ones((windowSize,windowSize))
kernel[(windowSize-1)/2,(windowSize-1)/2]=0

numIter=100

valueThreshold=0.375*((windowSize**2)-1) #Slightly xenophilic, 37.5% corresponds to a threshold of 3

def randomPopulationGrid():

	A=np.append(np.append( np.ones((1,numDem)), -1*np.ones((1,numRep))), np.zeros((1,numGrid-numDem-numRep)))
	
	np.random.shuffle(A)
	A=A.reshape((numRows,numColumns))

	return A

def visualMap(populationGrid):
	Map=np.copy(populationGrid)
	Map[np.where(Map==-1)]=0.5
	Map[np.where(Map==1)]=1
	
	return Map

populationGrid=randomPopulationGrid()
emptyHouses=np.asarray(np.asarray(np.where(populationGrid==0)).transpose())
print(np.shape(emptyHouses))

cv2.namedWindow('Population Grid')
cv2.namedWindow('Dem Value Grid')
cv2.namedWindow('Rep Value Grid')

for iter in range(0,numIter):
	print("Iteration "+ str(iter))

	populationGridOne=np.copy(populationGrid)
	populationGridOne[np.where(populationGridOne==-1)]=0	#Masking out opposition
	
	populationGridNegativeOne=np.copy(populationGrid)
	populationGridNegativeOne[np.where(populationGridNegativeOne==1)]=0	#Masking out opposition

	valueGridOne=signal.fftconvolve(populationGridOne, kernel, mode='same')#gives a map of the number of similar individuals -satisfaction
	valueGridNegativeOne=-1*signal.fftconvolve(populationGridNegativeOne, kernel, mode='same')#gives a map of the number of dissimilar individuals -satisfaction
		
	cv2.imshow('Dem Value Grid', (valueGridOne)/((windowSize**2)-1))
	cv2.imshow('Rep Value Grid', (valueGridNegativeOne)/((windowSize**2)-1))
	cv2.imshow('Population Grid', visualMap(populationGrid))
	
	cv2.waitKey(1)

	repopulationGrid=populationGrid

	if((iter%10)==0):
		cv2.imwrite('iteration'+str(iter)+'.bmp', visualMap(populationGrid)*(2**8))
	
	numSatisfied=0
	for i in range(0,numRows):
		for j in range(0,numColumns):

			if(repopulationGrid[i,j]==1):
				valueGrid=valueGridOne	
		
			if(repopulationGrid[i,j]==-1):
				valueGrid=valueGridNegativeOne
			
			if(repopulationGrid[i,j]==0 or valueGrid[i,j]>valueThreshold):
				numSatisfied+=1
				continue

			numSatisfied+=1
	
			emptyIndex=np.random.randint(0,numGrid-numDem-numRep)
			shiftIndex=emptyHouses[emptyIndex][:]

			repopulationGrid[i,j], repopulationGrid[shiftIndex[0],shiftIndex[1]]=repopulationGrid[shiftIndex[0],shiftIndex[1]], repopulationGrid[i,j]

			emptyHouses[0:-1,:]=np.append(emptyHouses[0:emptyIndex,:], emptyHouses[emptyIndex+1:,:], axis=0)
			emptyHouses[-1,:]=(np.array([i, j]))

			populationGrid=repopulationGrid

cv2.imwrite('iteration99.bmp', visualMap(populationGrid)*(2**8))

cv2.waitKey()
cv2.destroyAllWindows()
