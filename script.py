import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def calculateMeans(X,y):
    counts = {}
    means = np.zeros(shape=(len(X[0]), int(y.max())))
    for i, x in enumerate(X):
        label = int(y[i][0]) - 1
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
        for j, attr in enumerate(x):
            means[j][label] += attr
    for i, attrs in enumerate(means):
        for j, sums in enumerate(attrs):
            means[i][j] = sums / counts[j]
    return means

def calculateCovariance(X):
    d = len(X[0])

    attrMeans = np.mean(X, axis=0)
    variances = np.zeros(d)
    covariance = np.zeros(shape=(d,d))
    for i in range(0, d):
        for j in range(0, d):
            var_covar = 0
            summation = 0
            if i == j:
                for l in range(0, d):
                    summation += (X[l][i] - attrMeans[i]) ** 2
            else:
                for l in range(0, d):
                    summation += (X[l][i] - attrMeans[i]) * (X[l][j] - attrMeans[j])
            var_covar = summation / (d - 1)
            covariance[i][j] = var_covar
    return covariance

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 

    # IMPLEMENT THIS METHOD
    return calculateMeans(X,y),calculateCovariance(X)

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    dataByLabel = {}

    for i, x in enumerate(X):
        label = int(y[i][0]) - 1
        if label not in dataByLabel:
            dataByLabel[label] = []
        dataByLabel[label].append(x)

    covariances = []
    for label in sorted(dataByLabel.keys()):
        data = dataByLabel[label]
        covariances.append(calculateCovariance(data))

    return calculateMeans(X,y),covariances

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    predictions = np.zeros(shape=(len(ytest),1))
    siginv = np.linalg.inv(covmat)
    k = int(y.max())
    pi = []
    
    # calculate base probabilities
    for i in range(0, k):
        pi.append(0)
    for yi in ytest:
        pi[int(yi[0]) - 1] += 1
    for i, val in enumerate(pi):
        pi[i] = val / len(ytest)

    # Make predictions
    for i, x in enumerate(Xtest):
        discrimnants = []
        for classification in range(0, k):
            u_k = []
            for variable, value in enumerate(x):
                u_k.append(means[variable][classification])
            u_k = np.array(u_k)
            d_k = np.matmul(np.matmul(x.T, siginv), u_k) - (np.matmul(np.matmul(u_k.T, siginv), u_k) / 2) + pi[classification]
            discrimnants.append(d_k)
        predictions[i][0] = np.argmax(discrimnants) + 1

    # Calculate accuracies
    correct = 0
    for i, val in enumerate(predictions):
        if val[0] == ytest[i][0]:
            correct += 1
    acc = correct / len(ytest)
    return acc,predictions

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    predictions = np.zeros(shape=(len(ytest),1))
    k = int(y.max())
    pi = []
    
    # calculate base probabilities
    for i in range(0, k):
        pi.append(0)
    for yi in ytest:
        pi[int(yi[0]) - 1] += 1
    for i, val in enumerate(pi):
        pi[i] = val / len(ytest)

    # Make predictions
    for i, x in enumerate(Xtest):
        discrimnants = []
        for classification in range(0, k):
            sigma = covmats[classification]
            siginv = np.linalg.inv(sigma)

            u_k = []
            for variable, value in enumerate(x):
                u_k.append(means[variable][classification])
            u_k = np.array(u_k)

            d_k =  pi[classification] + np.matmul(np.matmul(x.T, siginv), u_k) - (np.matmul(np.matmul(u_k.T, siginv), u_k) / 2) - (np.matmul(np.matmul(x.T, siginv), x) / 2) - (np.log(np.linalg.det(sigma)) / 2)
            discrimnants.append(d_k)

        predictions[i][0] = np.argmax(discrimnants) + 1

    # Calculate accuracies
    correct = 0
    for i, val in enumerate(predictions):
        if val[0] == ytest[i][0]:
            correct += 1
    acc = correct / len(ytest)
    return acc,predictions

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    # print(X,y)
    # IMPLEMENT THIS METHOD
    w = np.matmul(np.matmul(inv(np.matmul(X.T, X)), X.T), y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD      
    # print("lambd", lambd)
    # w = np.matmul(np.matmul( lambd * np.identity(len(X[0])) + inv(np.matmul(X.T, X)), X.T), y)
    w = np.matmul( np.matmul( inv( lambd * np.identity(len(X[0])) + np.matmul(X.T, X) ), X.T), y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    N = len(Xtest)
    mse = 0
    for i in range(0,N):
        mse += (y[i] - np.matmul(w.T, Xtest[i])) ** 2

    # mse /= N
    mse /= 2
    # print("MSE,", mse)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    N = len(X)
    error = 0
    for i in range(0,N):
        error += (y[i] - np.matmul(w.T, X[i])) ** 2
    error /= 2

    # −XT (Y − Xw) + λw 
    # print(np.matmul(-(X.T), (y - np.matmul(X, w))))
    d = len(w)
    error_grad = []
    for j in range(0, d):
        errorgrad_j = 0
        for i in range(0,N):
            errorgrad_j += np.multiply(np.matmul(w.T, X[i]) - y[i], X[i][j]) + (lambd * w[j])
        error_grad.append(errorgrad_j)
    error_grad = np.array(error_grad).flatten()

    # error_grad = np.matmul(-(X.T), (y - np.matmul(X, w))) + lambd * w
    # w − lambd X.T(Xw − Y)
    # error_grad = lambd * np.matmul(X.T, np.matmul(X, w.T) - y)

    # print(error_grad)
    # print("ERROR VALUE-----------------------------------", error[0])
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    
    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    print("Lambda is", lambd)
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    print("w_l is", w_l)
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
