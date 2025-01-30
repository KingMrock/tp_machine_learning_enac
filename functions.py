################################
######## Dataset Generation ######
#################################

import numpy as np
import matplotlib.pyplot as plt
import math as math
import sklearn as sk
from sklearn.datasets import make_regression

def generate_dataset(n,d):
    X,Y = make_regression(n, d, n_informative=100, bias=2, coef=False, noise=10.0,random_state=1)
    return X,Y

################################
######## Basic Gradient ########
#################################

def full_objective(Y, A, x):
    """
    Compute the objective function for the entire dataset.
    """
    return np.linalg.norm(Y - A.dot(x))**2


def gradient_regression(Y, A, x):
    """
    Compute the gradient of the regression function for the entire dataset.
    
    :param Y: a numpy array of shape (n)
    :param A: a numpy array of shape (n, d)
    :param x: compute the gradient at these parameters, numpy array of shape (d)
    
    :return: gradient: numpy array of shape (d)
    """
    
    n = len(Y)
    grad = -1/n * A.T @ (Y - A @ x)
    return grad

def gradient_descent(
        Y, 
        A, 
        initial_x, 
        nmax, lr,
        print_output=False):
    """
    Gradient Descent for Linear Least Squares problems.
    
    :param Y: numpy array of size (n)
    :param A: numpy array of size (n, d)
    :param initial_x: starting parameters, a numpy array of size (d)
    :param nmax: integer, number of iterations
    :param lr: learning rate=step size
    
    
    :return:
    - objectives, a list of loss values on the whole dataset, collected at the end of each pass over the dataset (epoch)
    - param_states, a list of parameter vectors, collected at the end of each pass over the dataset
    """
    xs = [initial_x]  # parameters after each update 
    objectives = []  # loss values after each update
    x = initial_x
    
    for epoch in range(nmax):
        grad = gradient_regression(Y, A, x)
        # update x through the gradient update
        x = x - lr * grad
    
        # store x and objective
        xs.append(x.copy())
        objective = full_objective(Y, A, x)
        objectives.append(objective)
        if print_output:
            print("GD({ep:04d}/{bi:04d}/{ti:04d}): objective = {l:10.2f}".format(ep=epoch,
                          bi=epoch, ti=len(Y) - 1, l=objective))
    return objectives, xs

#####################################
######## Accelerated Gradient ########
#####################################

def accelerated_gradient_descent(
        Y, 
        A, 
        initial_x,
        nmax,
        lr=-1,
        print_output=False):
    """
    Gradient Descent for Linear Least Squares problems.
    
    :param Y: numpy array of size (n)
    :param A: numpy array of size (n, d)
    :param initial_x: starting parameters, a numpy array of size (d)
    :param nmax: integer, number of iterations
    
    
    :return:
    - objectives, a list of loss values on the whole dataset, collected at the end of each pass over the dataset (epoch)
    - param_states, a list of parameter vectors, collected at the end of each pass over the dataset
    """
    xs = [initial_x]  # parameters after each update 
    objectives = []  # loss values after each update
    
    xs = [initial_x]  # parameters after each update 
    objectives = []  # loss values after each update
    
    # Compute L and mu
    eigvals = np.linalg.eigvalsh(A.T @ A / len(Y))
    if (lr == -1):
        L = eigvals.max()
    else:
        L = lr
    
    mu = eigvals.min()
    
    # Compute alpha and beta
    alpha = 4 / (np.sqrt(L) + np.sqrt(mu))**2
    beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    
    x = initial_x
    x_prev = initial_x
    
    for epoch in range(nmax):
        grad = -1 / len(Y) * A.T @ (Y - A @ x)
        
        # update x through the accelerated gradient update
        x_new = x - alpha * grad + beta * (x - x_prev)
        
        # store x and objective
        xs.append(x_new.copy())
        objective = np.linalg.norm(Y - A.dot(x_new))**2
        objectives.append(objective)
        
        if print_output:
            print("AGD({ep:04d}/{bi:04d}/{ti:04d}): objective = {l:10.2f}".format(ep=epoch,
                          bi=epoch, ti=len(Y) - 1, l=objective))
        
        x_prev = x
        x = x_new
    
    return objectives, xs


#####################################
######## Subgradient Descent ########
#####################################

def reg_objective(Y, A, x,lbda):
    # Compute the least squares objective over the whole dataset
    return np.linalg.norm(Y - A.dot(x))**2 + lbda*np.linalg.norm(x=x,ord=1)

def subgradient_LASSO(Y, A, x, lbda):
    """
    Compute a subgradient of the LASSO problem for the entire dataset.
    
    :param Y: a numpy array of shape (n)
    :param A: a numpy array of shape (n, d)
    :param x: compute the gradient at these parameters, numpy array of shape (d)
    :param lbda: a positive scalar
    
    :return: subgradient: numpy array of shape (d)
    """

    n = len(Y)
    grad_least_squares = -1.0/n * A.T @ (Y - A @ x)
    
    # Subgradient for L1 regularization
    subgrad_l1 = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            subgrad_l1[i] = 1
        elif x[i] < 0:
            subgrad_l1[i] = -1
        else:
            subgrad_l1[i] = 0  # Could be any value in [-1,1] if x[i] == 0
    
    return grad_least_squares + lbda * subgrad_l1


def subgradient_descent(
        Y, 
        A, 
        initial_x, 
        nmax, lr, lbda,
        print_output=False):
    """
    Subgradient Descent for Linear Least Squares problems.
    
    :param Y: numpy array of size (n)
    :param A: numpy array of size (n, d)
    :param initial_x: starting parameters, a numpy array of size (d)
    :param nmax: integer, number of iterations
    :param lr: learning rate=step size
    :param lbda: a positive scalar
    
    :return:
    - objectives, a list of loss values on the whole dataset, collected at the end of each pass over the dataset (epoch)
    - param_states, a list of parameter vectors, collected at the end of each pass over the dataset
    """
    xs = [initial_x]  # parameters after each update 
    objectives = []  # loss values after each update
    x = initial_x
    best_obj=reg_objective(Y,A,initial_x,lbda)
    
    for epoch in range(nmax):
        subgrad = subgradient_LASSO(Y, A, x, lbda)

        
        x = x - lr * subgrad
    
        # store x and objective
        xs.append(x.copy())
        objective = reg_objective(Y, A, x,lbda)
        if objective<best_obj:
            best_obj=objective           
        objectives.append(best_obj)
        if print_output:
            print("GD({ep:04d}/{bi:04d}/{ti:04d}): objective = {l:10.2f}".format(ep=epoch,
                          bi=epoch, ti=len(Y) - 1, l=best_obj))
    return objectives, xs



#####################################
######## Proximal Gradient ##########
#####################################

def prox_L1(v,alpha):
    """
    Proximal operator for the L1 norm (soft thresholding).
    """
    return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)

def prox_gradient_descent(
        Y, 
        A, 
        initial_x, 
        nmax, lr,lbda,
        print_output=False):
    """
    Gradient Descent for Linear Least Squares problems.
    
    :param Y: numpy array of size (n)
    :param A: numpy array of size (n, d)
    :param initial_x: starting parameters, a numpy array of size (d)
    :param nmax: integer, number of iterations
    :param lr: learning rate=step size
    :param lbda positive scalar
    
    
    :return:
    - objectives, a list of loss values on the whole dataset, collected at the end of each pass over the dataset (epoch)
    - param_states, a list of parameter vectors, collected at the end of each pass over the dataset
    """
    xs = [initial_x]  # parameters after each update 
    objectives = []  # loss values after each update
    x = initial_x
    
    for epoch in range(nmax):
        grad = gradient_regression(Y, A, x)
        # update x through the gradient update
        x = prox_L1(x - lr * grad, lr*lbda)
    
        # store x and objective
        xs.append(x.copy())
        objective = reg_objective(Y, A, x,lbda)
        objectives.append(objective)
        if print_output:
            print("GD({ep:04d}/{bi:04d}/{ti:04d}): objective = {l:10.2f}".format(ep=epoch,
                          bi=epoch, ti=len(Y) - 1, l=objective))
    return objectives, xs



#####################################
######## Stochastic Gradient ########
#####################################

def minibatch_gradient(Y_batch, A_batch, x):
    """
    Compute a mini-batch stochastic gradient from a subset of `num_examples` from the dataset.
    
    :param Y_batch: a numpy array of shape (batch_size)
    :param A_batch: a numpy array of shape (batch_size, d)
    :param x: compute the mini-batch gradient at these parameters, numpy array of shape (d)
    
    :return: stoc_grad: numpy array of shape (d)
    """
    
    stoc_grad = -1/len(Y_batch) * A_batch.T @ (Y_batch - A_batch @ x)
    return stoc_grad



def stoc_gradient_descent(
        Y, 
        A, 
        initial_x, 
        nmax, lr, batch_size,
        print_output=False):
    """
    Stochastic gradient descent for Linear Least Squares problems.
    
    :param Y: numpy array of size (n)
    :param A: numpy array of size (n, d)
    :param initial_x: starting parameters, a numpy array of size (d)
    :param nmax: integer, number of iterations
    :param lr: learning rate=step size
    :param batch_size: number of samples used for calculating the stochastic gradient 
    
    
    :return:
    - objectives, a list of loss values on the whole dataset, collected at the end of each pass over the dataset (epoch)
    - param_states, a list of parameter vectors, collected at the end of each pass over the dataset
    """
    xs = [initial_x]  # parameters after each update 
    objectives = []  # loss values after each update
    x = initial_x
    
    for epoch in range(nmax):
        n=len(Y)
        #randomly get a batch of data
        indices = np.random.choice(n, batch_size, replace=False)
        Y_batch = Y[indices]
        A_batch = A[indices]


        grad = minibatch_gradient(Y_batch, A_batch, x)
        # update x through the gradient update
        x = x - lr * grad
        lr = lr * 0.99
        
            
        # store x and objective
        xs.append(x.copy())
        objective = full_objective(Y, A, x)
        objectives.append(objective)
        if print_output:
            print("SGD({ep:04d}/{bi:04d}/{ti:04d}): objective = {l:10.2f}".format(ep=epoch,
                          bi=epoch, ti=len(Y) - 1, l=objective))
    return objectives, xs



#####################################
######## SVM (other part)  ##########
#####################################

from sklearn.utils import shuffle

def generatedata2(n):
    # n must be an even integer
    np.random.seed(0)
    mean1, cov1, n1 = [1, 5], [[1,1],[1,2]], int(n/2)  # n/2 samples of class 1
    x1 = np.random.multivariate_normal(mean1, cov1, n1)
    y1 = np.ones(n1, dtype=int)
    mean2, cov2, n2 = [2.5, 2.5], [[1,0],[0,1]], int(n/2) # n/2 samples of class -1
    x2 = np.random.multivariate_normal(mean2, cov2, n2)
    y2 = -np.ones(n2, dtype=int)
    x = np.concatenate((x1, x2), axis=0) # concatenate the samples
    y = np.concatenate((y1, y2))
    x,y = shuffle(x,y)
    
    return [x,y]

def plotdata(x,y):
    for i in range(x.shape[0]):
        if y[i]==-1:
            cl='bo'
        else:
                cl='r+'
        plt.plot(x[i,0],x[i,1],cl);
    return

def plothyperplane(w,b):
    xx=np.linspace(x[:,0].min(),x[:,0].max(),100)
    #print(xx)
    yy=(-b-w[0]*xx)/w[1]
    yy1=yy+1/np.linalg.norm(w)
    yy2=yy-1/np.linalg.norm(w)
    #print(yy)
    plt.plot(xx,yy,'k-')
    plt.plot(xx,yy1,'g-')
    plt.plot(xx,yy2,'g-')
    return

def SVMobjective(x, y, d, C, theta):
    w = theta[:d]
    b = theta[d]
    hinge_losses = np.maximum(0, 1 - y * (x @ w + b))
    return 0.5 * np.linalg.norm(w)**2 + C * np.sum(hinge_losses)


def minibatch_subgrad_svm(x_batch,y_batch,d,C,theta):
    w = theta[:d]
    b = theta[d]
    margins = y_batch * (x_batch @ w + b)
    mask = margins < 1
    subgrad_w = w - C * np.mean(y_batch[mask, np.newaxis] * x_batch[mask], axis=0)
    subgrad_b = -C * np.mean(y_batch[mask]) if np.any(mask) else 0
    subgrad = np.concatenate([subgrad_w, [subgrad_b]])
    return subgrad


def svm_SGD(x,y,d,C,theta,batch_size,lr,nmax, print_output=False):
    """
        Parameters:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        d (int): Dimensionality of the feature space.
        C (float): Regularization parameter.
        theta (numpy.ndarray): Initial parameter vector.
        batch_size (int): Size of the mini-batches for SGD.
        lr (float): Learning rate for the gradient descent.
        nmax (int): Number of epochs for the SGD.
        Returns:
        objectives (list): List of objective values at each iteration.
        thetas (list): List of parameter vectors at each iteration.
    
    Stochastic gradient descent for SVM problems.
    """ 

    
    thetas = [theta]
    objectives = []
    for epoch in range(nmax):
        for iter in range(int(len(y)/batch_size)):
            indices = np.random.choice(n, batch_size, replace=True)
            SVMgrad=minibatch_subgrad_svm(x[indices],y[indices],d,C,theta)
        
            # update x through the gradient update
            theta -= lr * SVMgrad
                
            # store x and objective
            thetas.append(theta.copy())
            objective=SVMobjective(x,y,d,C,theta)
            objectives.append(objective)
            if print_output:
                print("SGD({ep:04d}/{bi:04d}/{ti:04d}): objective = {l:10.2f}".format(ep=epoch,
                              bi=epoch, ti=len(y) - 1, l=objective))
    return objectives, thetas
