import numpy as np
import pandas as pd
from time import perf_counter 
from collections import Counter
### -------------------------------------------------------------------------------------------------
### -------------------The 4 required functions are below loadData func------------------------------
### -------------------------------------------------------------------------------------------------
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
### -------------------------------------------------------------------------------------------------
def dataNorm(X):
    '''
    Normalization of the feature data points. 
    Inputs: raw data set from abalone.data file. Type is numpy 2D array.
    Outputs: a normalized dataset. Type is numpy 2D array.
    '''
    xMerged = X[:,-1].T #output col. switch it's column to a row for vstack later    
    f_transpose = X[:, :8].T #feature cols. switch the columns to rows for iteration later
    
    for i in f_transpose:  
        arr_transpose = (i - np.min(i)) / np.ptp(i) # normalization equation
        xMerged = np.vstack((xMerged, arr_transpose)) # after normalization, stack the normalized rows together
        
    y_output = xMerged[0] # a row of 'Rings' data points
    final_merged = np.vstack((xMerged[1:], y_output)) # vstack the 8 features and the output at the bottom of the stack 
    return final_merged.T # transpose back to the initial form where columns are features, rows are datapoints
### -------------------------------------------------------------------------------------------------
def splitTT(X_norm, PercentTrain):
    '''
    Function task is to split the normalized dataset into a train set and a test set.
    Inputs: a normalized data set of type numpy 2d array, a ratio of train-to-test split.
    Output: a list contains a train set and test set. Both are of numpy 2D arrays type.
    '''
    
    np.random.shuffle(X_norm) # shuffles the rows in the X_norm matrix
    row_num = X_norm.shape[0] # get the num of rows
    ratio = int(row_num*PercentTrain) # ratio expresses in int num
    X_test = X_norm[ratio:,:]
    X_train =  X_norm[:ratio,:]
    X_split = [X_train, X_test] # return a list of X train and X test sets
    return X_split
### -------------------------------------------------------------------------------------------------
def splitCV(X_norm, K): # Split a dataset into k folds
    '''
    Partition the data into K parts. 
    One part become test and the rest become train in a K iteration process.
    Inputs: a normalized data set of type numpy 2D array and an integer K .
    that determines the total parts to split into and iterates K times. 
    Output: a list contains k elements where each is of type numpy 2D array.
    '''
    dataset_split = []
    np.random.shuffle(X_norm) # shuffles the rows in the X_norm matrix
    fold_size = int(len(X_norm) / K) # compute the num of rows per fold

    for i in range(K):
        if i == K-1:
            dataset_split.append(X_norm)
        else:
            dataset_split.append(X_norm[:fold_size])
            X_norm = X_norm[fold_size:]       
    return dataset_split
### -------------------------------------------------------------------------------------------------
def KNN(xy_train, xy_test, k): # inputs = X_train, test_instance = row, k = number of neighbors  ### 2ND ITERATION
    '''
    Calculate k-nn for given k and return the accuracy value 
    Input: a train set and a test set of type 2D numpy array.
    Input: an integer k which is the number of neighbors.
    Output: an accuracy value of type float.   
    '''   
    def predict_instance_numpy(inputs, labels, test_instance, k):  # inputs = X_train, labels=y_train, test_instance = row, k= number of neighbors
        inputs['distance'] = np.linalg.norm(inputs.values - test_instance.values, axis=1)  # calculate L2 norm between all training points and given test_point 
        inputs = pd.concat([inputs, labels], axis=1)               # concatenate inputs and labels (y_train) before sorting the distances
        inputs = inputs.sort_values('distance', ascending=True)    # sort based on distance
        neighbors = inputs.head(k)               # pick k neighbors
        classes = neighbors.iloc[:, -1].tolist() # get list from dataframe column
        majority_count = Counter(classes) # create counter of labels
        return majority_count.most_common(1).pop()[0]

    X_train  = pd.DataFrame(xy_train[:, :-1])
    y_train = pd.DataFrame(xy_train[:, -1])
    X_test = pd.DataFrame(xy_test[:, :-1])
    y_test = pd.DataFrame(xy_test[:, -1])
#     print(f'y_test shape:{y_test.shape}')
    predictions = np.zeros(X_test.shape[0])
    X_test.reset_index(drop=True, inplace=True)
    
    for index, row in X_test.iterrows():
        predictions[index] = predict_instance_numpy(X_train.copy(), y_train.copy(), row, k)
#     print(f'predictions mean:\n{np.mean(predictions)}')
#     print(f'predictions shape: {predictions.shape}')
    true_values = y_test.to_numpy()
    true_val_list = []
    
    for i in true_values:
        true_val_list.append(i[0])

    true_val_np = np.array(true_val_list)
    accuracy = np.mean(predictions == true_val_np) # take the mean tally of all the data points that's True
    return accuracy
### -------------------------------------------------------------------------------------------------
### -------------------The 4 required functions are listed above-------------------------------------
### -------------------------------------------------------------------------------------------------
### -------------------------------------------------------------------------------------------------
### ----The calling funcs and other funcs to answer other Questions are below -----------------------
### -------------------------------------------------------------------------------------------------
def knnMain(filename, percentTrain, k):
    '''
    a main calling function responsible to call loadData, dataNorm, splitTT and KNN functions.
    Inputs: file name(.data), train-test_split ratio (float) and number of nearest neighbour (integer)
    Output: an accuracy value of type float.
    ''' 
    X = loadData(filename) # data load
    X_norm = dataNorm(X) # normalization
    X_split = splitTT(X_norm,percentTrain)  # split the data split to get [X_train, X_test]    
    accuracy = KNN(X_split[0],X_split[1], k)  # X_split[0] is Xy_train, X_split[1] is Xy_test    
    return accuracy
### -------------------------------------------------------------------------------------------------
## for-loop block code to call the knnMain to perform train_test split 
TT_accuracy_list = []
TT_time_list = []
for neighbor in [1,5,10,15,20]:  # looping over number of k-neighbors
    temp_accuracy = []
    temp_time = []
    for train_portion in [0.7,0.6,0.5]: # Looping over train_test_split_ratio
        t1_start = perf_counter() # Start the stopwatch / counter 
        acc = knnMain('abalone.data', train_portion, neighbor)
        print(f"Neighbor: {neighbor}\tTrain_portion: {train_portion}\tAccuracy: {acc}")
        t1_stop = perf_counter() # Stop the stopwatch / counter 
        print(f'Elapsed time taken {t1_stop-t1_start} seconds\n') 
        temp_accuracy.append(acc)
        temp_time.append(t1_stop-t1_start)
    TT_accuracy_list.append(temp_accuracy)
    TT_time_list.append(temp_time)
print('Run completed.')
### -------------------------------------------------------------------------------------------------
def knnCV_Main(filename, cv_num, k_neighbors): # k = number of neighbors
    '''
    a main calling function responsible to call loadData, dataNorm, splitCV and KNN functions.
    Inputs: file name(.data), cv number (integer) and number of nearest neighbour (integer)
    Output: an average accuracy value of type float.   
    '''
    accuracy = []
    X = loadData(filename) # data load
    X_norm = dataNorm(X)   # normalization
    X_cv = splitCV(X_norm, cv_num) # split the data set into K folds = number of parts
    accuracy_sum = 0
    print('\nCV_computation ongoing ... ')
    for idx, list_array in enumerate(X_cv): # looping the dataset for cross validation 
        duplicate = X_cv.copy()
        test = list_array
        del duplicate[idx]  # delete the test element from the duplicate set, remaining elements become train elements
        train = duplicate
        train = np.vstack((train))
        accuracy = KNN(train, test, k_neighbors)  # X_split[0] is Xy_train, X_split[1] is Xy_test
        accuracy_sum += accuracy
    return accuracy_sum/cv_num  # return the average accuracy of the k number of neighbors
### -------------------------------------------------------------------------------------------------
## for-loop block code to call the knnCV_Main to perform cross validation 
CV_accuracy_list = []
CV_time_list = []
for neighbor in [1,5, 10, 15, 20]:  # looping over number of k-neighbors
    temp_CV_accuracy = []
    temp_CV_time = []
    for cv in [5,10, 15]:  # Looping over the cv numbers
        t1_start = perf_counter() # Start the stopwatch / counter 
        acc_CV = knnCV_Main('abalone.data', cv, neighbor)
        print(f"Neighbor: {neighbor}\tCV: {cv}\tAccuracy: {acc_CV}")
        t1_stop = perf_counter() # Stop the stopwatch / counter 
        print(f'Elapsed time taken {t1_stop-t1_start} seconds\n') 
        temp_CV_accuracy.append(acc_CV)
        temp_CV_time.append(t1_stop-t1_start)
    CV_accuracy_list.append(temp_CV_accuracy)
    CV_time_list.append(temp_CV_time)
print('Run completed.')
### -------------------------------------------------------------------------------------------------
