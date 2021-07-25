import numpy as np

def loadData(filename):
  # load data from filename into X
  X=[]
  count = 0
  
  text_file = open(filename, "r")
  lines = text_file.readlines()
    
  for line in lines:
    X.append([])
    words = line.split(",")
    # convert value of first attribute into float  
    for word in words:
      if (word=='M'):
        word = 0.333
      if (word=='F'):
        word = 0.666
      if (word=='I'):
        word = 1
      X[count].append(float(word))
    count += 1
  
  return np.asarray(X)

def testNorm([X_norm]):
  xMerged = np.copy(X_norm[0])
  #merge datasets
  for i in range(len(X_norm)-1):
    xMerged = np.concatenate((xMerged,X_norm[i+1]))
  print np.mean(xMerged,axis=0)
  print np.sum(xMerged,axis=0)

# this is an example main for KNN with train-and-test + euclidean
def knnMain(filename,percentTrain,k):
 
  #data load
  X = loadData(filename)
  #normalization
  X_norm = dataNorm(X)
  #data split: train-and-test
  X_split = splitTT(X_norm,percentTrain)
  #KNN: euclidean
  accuracy = knn(X_split[0],X_split[1],k)
  
  return accuracy
  