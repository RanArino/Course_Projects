import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib as plt



class Standardization:
    def __init__(self):
        #initialize all variables
        self.mu_old = 0
        self.sd_old = 0
        self.mu = 0
        self.sd = 0
        self.t = 0
        self.X = np.array([])
        self.X_ss = np.array([])

    def fit_transform(self, t: int, X: NDArray):
        """
        Standarize X features by mean and std of the time periods of "t", then return the standardized features.
        
        Parameters:
        "t": observing time ranges or number of data; a window of data.
        "X": the matrix of independent variables.
        """
        # store all given parameters
        self.t = t
        self.X = X
        self.X_ss = np.zeros(X.shape)

        # initial standardization
        self.mu, self.sd = np.mean(X[0:t], axis=0), np.std(X[0:t], axis=0)
        self.X_ss[0:t] = (X[0:t] - self.mu) / self.sd

        # standarize new feature one by one
        for i, idx in enumerate(range(t, len(X))):
            # assign a series of newly standardized features
            self.X_ss[idx] = self.update(x_del=X[i], x_new=X[idx])

        return self.X_ss
        

    def transform(self, x: NDArray):
        """
        Return value is the same as self.update().
        A given series of features ("x") is added to the stored feature matrix.

        This function is invoked everytime when a new series of new features.
        Make sure the consistency of the time series data. 

        Parameter:
        - "x": a series of newly updated features and has never been add in self.X 
        """
        # a series of deleted features 
        x_del = self.X[-self.t]
        # update X
        self.X = np.append(self.X, x)
        # assign standardized features
        new_features = self.update(x_del=x_del, x_new=x)
        self.X_ss = np.append(self.X_ss, new_features)

        return new_features


    def update(self, x_del: NDArray, x_new: NDArray):
        """
        Return a series of standardized features one by one based on the mean and std of the most recent "self.t" number of data.
        In other words, a series of new features are standardized by the moving average and std over the "self.t" periods.

        Paratermers:
        - "x_del": a series of unscoped features.
        - "x_new": a series of the most recent features.
        """
        # assign current mean and std as old ones
        self.mu_old, self.sd_old = self.mu, self.sd
        # newly deleted features will be at -120 from the current feature matrix
        self.mu = self.mu_old + (x_new - x_del) / self.t
        self.sd = np.sqrt(((self.t-1)*self.sd_old**2 - (x_del - self.mu_old)**2 + (x_new - self.mu)**2) / (self.t-1))
        # assign standardized features
        new_features = (x_new - self.mu) / self.sd

        return new_features
    
def gradient_descent(X, y, t, b, alpha=0.01, lambda_=0.5):
    """
    Return the following three matrix (dtype: np.array)
    - "theta"  -> parameters (intercept + coefficients) at each step
    - "y_hats" -> predicted values at each step
    - "sigma"  -> prediction errors (actual - predicted values); SSE

    Parameters:
    - "X": np.array -> independent variables
    - "y": np.array -> target variables
    - "t": int -> number of data that were used for the initial parameter creation.
    - "b": int -> mini-batch sizes(b <= t) 
    - "alpha": learning rate for gradient descent
    - "lambda_": hyperparameter for the elastic net regularization.

    Brief Steps:
    - Initialize matries for theta, y_hats, sigma.
    - Apply a given number of data ("t") to the mutiple linear regression (normal equation).
    - Define the initial parameters from the trained model.
    - At each step (total steps are len(y) - t):
        - Get a single pair of unfamilar data; both X and y.
        - Predict the target ("y_hats") based on the latest parameters("theta[i]").
        - Calculate the difference between actual and predicted values; "sigma[i]".
        - Update parameters for the next step ("theta[i+1]") by a given number of the errors;
           if b=10, recalculating the predicted error based on the theta[i] and the latest 10 data.
    """
    # define all matrix to be returned
    theta = np.zeros((len(y)-t + 1, 6))
    y_hats = np.zeros((len(y)-t, 1))
    sigma = np.zeros((len(y)-t, 1))
    # Modify the matrix of features; adding bias 
    X = np.insert(X[:], 0, 1, axis=1)
    
    # initial training
    init_X, init_y = X[0:t], y[0:t]
    # obtain the initial parameters based on normal equation form
    theta[0] = np.linalg.inv(init_X.T.dot(init_X)).dot(init_X.T).dot(init_y).reshape(1,-1)
    
    # incremental learnings
    for i, idx in enumerate(range(t, len(y))):
        # get new data (X_i includes bias)
        X_i, y_i = X[idx] , y[idx]
        # predicted variable
        y_hats[i] = np.dot(X_i, theta[i])
        # predicted error
        sigma[i] =  (y_hats[i] - y_i)
        # given number of predicted errros based on the current parameters
        if b > 1:
            # the given number (batch size; "b") of the latest X and y
            X_b, y_b = X[idx-b+1:idx+1], y[idx-b+1:idx+1]  
            # predicted error in the latest number "b" of data based on current parameters
            sigmas = np.dot(X_b, theta[i].reshape(-1, 1)) - y_b
            # calculate the mean of error weighted features (ewf_mu)
            ewf_mu = np.mean(X_b * sigmas, axis=0)

        else:
            # if batch size is 1, just consider a predicted error based on the most recent data
            ewf_mu = X_i * sigma[i]
        
        # get the partial derivative 
        # pd = 2 / b * np.
        # the gradient of the loss function with respect to the new observation 
        gradient = 2 * (ewf_mu + (lambda_*alpha*np.sign(theta[i]) + 2*(1-lambda_)*alpha*theta[i]))
        # update all parameters by stochastic gradient descent
        theta[i+1] = theta[i] - alpha * gradient

    return theta, y_hats, sigma

def evaluation(X, y, t, theta, y_hats, sigma):
    """
    Return the data frame; the following measureas by brackward eliminations:
    - Root Mean Square Error (rmse)
    - Standatd Error of Estimate (se)
    - Coefficient of Determination (r2)
    - Adjusted Coefficient of Determination (adj_r2)

    Parameters:
    - "t": number of months that were used for the initial training data.
    - "X": independent variables
    - "y": the actual target value (whole period)
    - "theta": all parameters (intercept & coefficients) at each step
    - "y_hats": the predicted target value at each step
    - "sigma": difference between actual and predicted values at each step

    Requirement: len(y[t:]) == len(y_hats)
    """
    # (0): Define Variables
    #  number of observations and features (excluding bias)
    n, k = len(y_hats), len(theta[0]) - 1
    #  measures matrix
    mm = np.zeros((k+1, 4))
    # assign the data frame index and columns
    rows = ['original'] + [f'theta{i+1}=0' for i in range(k)]
    cols = ['rmse', 'se', 'r2', 'adj_r2']
    #  mean of actual "y" over the "t" months
    #y_mean = np.array([y[i:i+t].mean() for i in range(len(y)-t)]).reshape(-1, 1)
    y_mean = np.mean(y[t:])
    #  sum of square total
    sst = np.sum((y[t:] - y_mean)**2)
    #  define feature matrix and target based on "t"
    X_, y_ = np.insert(X[t:n+t], 0, 1, axis=1), y[t:]

    #  number of coefficients
    for i in range(k+1):
        # simply applying the given sigma
        if i == 0:
            sigma_ = sigma
        # conduct the backward elimination
        else:
            # copy the parameters matrix
            theta_ = theta[:n].copy()
            # change a particular coefficient to 0 arbitrarily.
            theta_[:, i] = 0
            # based on revised parameters, get the predicted value
            y_hats_ = np.sum(X_ * theta_, axis=1).reshape(-1, 1)
            # predicted error
            sigma_ = y_ - y_hats_    
    
        # Calculate Measures (rmse, se, r2, adj_r2, in order)
        sse = np.sum(sigma_**2)
        mm[i, 0] = np.sqrt((sigma_**2).mean()) 
        mm[i, 1] = np.sqrt(sse / (n - k - 1))
        mm[i, 2] = 1 - sse/sst
        mm[i, 3] = 1 - (sse/(n - k -1))/(sst/(n-1))
         
    return pd.DataFrame(data=mm, index=rows, columns=cols)


def online_learning(t, X, y, alpha=0.01, lambda_=0.5):
    """
    Return the following three matrix (dtype: np.array)
    - "theta"  -> parameters (intercept + coefficients) at each step
    - "y_hats" -> predicted values at each step
    - "sigma"  -> prediction errors (actual - predicted values); SSE

    Parameters:
    - "t": int -> number of months that were used for the initial training data.
    - "X": np.array -> independent variables
    - "y": np.array -> target variables
    - "alpha": learning rate for gradient descent
    - "lambda_": hyperparameter for the elastic net regularization.

    Brief Steps:
    - Initialize matries for theta, y_hats, sigma.
    - Apply a given time ranges ("t") of the dataset to the mutiple linear regression.
    - Define the initial parameters from the trained model.
    - For each new data point (incremental learning):
        - Predict the target; "y_hats".
        - Calculate the difference between actual and predicted values; "sigma"
        - Update parameters by (stochastic) gradient descent with elastic regularization; "theta"
    """
    # define all matrix to be returned
    theta = np.zeros((len(y)-t + 1, 6))
    y_hats = np.zeros((len(y)-t, 1))
    sigma = np.zeros((len(y)-t, 1))
    # Modify the matrix of features; adding bias 
    X = np.insert(X[:], 0, 1, axis=1)
    
    # initial training
    init_X, init_y = X[0:t], y[0:t]
    # obtain the initial parameters based on normal equation form
    theta[0] = np.linalg.inv(init_X.T.dot(init_X)).dot(init_X.T).dot(init_y).reshape(1,-1)
    
    # incremental learnings
    for i, idx in enumerate(range(t, len(y))):
        # get new data (X_i includes bias)
        X_i, y_i = X[idx] , y[idx]
        # predicted variable
        y_hats[i] = np.dot(X_i, theta[i])
        # predicted error
        sigma[i] =  (y_i - y_hats[i])
        # the gradient of the loss function with respect to the new observation (normal: X_i * sigma[i])
        gradient = -2 * X_i * sigma[i] + 2 * lambda_ * theta[i] + (1-lambda_) * np.sign(theta[i])
        # update all parameters by stochastic gradient descent
        theta[i+1] = theta[i] - alpha * gradient

    return theta, y_hats, sigma

def batch_learning(t, X, y, labmda_=0.5):
    """
    Return the following three matrix (dtype: np.array)
    - "theta"  -> parameters (intercept + coefficients) at each step
    - "y_hats" -> predicted values at each step
    - "sigma"  -> prediction errors

    Parameters:
    - "t": int -> number of months that were used for the initial training data.
    - "X": np.array -> independent variables
    - "y": np.array -> target variables
    - "lambda_": hyperparameter for the elastic net regularization.

    Brief Steps:
    - Initialize matries for theta, y_hats, sigma.
    - Train the model with a given time ranges ("t") of the dataset.
    - For each new data point (batch learning):
        - get all parameters (intercept and coefficients) by the least square method; "theta"
        - Predict the target value; "y_hats".
        - Calculate the difference between actual and predicted values; "sigma"
    """
    # define all matrix to be returned
    theta = np.zeros((len(y)-t + 1, 6))
    y_hats = np.zeros((len(y)-t, 1))
    sigma = np.zeros((len(y)-t, 1))

    # batch learning
    for i, idx in enumerate(range(t, len(y))):
        # get the recent "t" months data (X_i includes bias)
        X_i, y_i = np.insert(X[i:idx], 0, 1, axis=1) , y[i:idx]
        # fit the training data (normal equation form)
        theta[i] = np.linalg.inv(X_i.T.dot(X_i)).dot(X_i.T).dot(y_i).reshape(1,-1)
        # predicted value
        X_new = np.append(np.array([1]), X[idx])
        y_hats[i] = np.dot(theta[i], X_new)
        # predicted error
        sigma[i] = y[idx] - y_hats[i]

    # trained the most recent data
    last = len(y)
    X_i, y_i = np.insert(X[last-t:last], 0, 1, axis=1) , y[last-t:last]
    theta[last-t] = np.linalg.inv(X_i.T.dot(X_i)).dot(X_i.T).dot(y_i).reshape(1,-1)

    return theta, y_hats, sigma