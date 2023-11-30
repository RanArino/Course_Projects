import numpy as np
from numpy.typing import NDArray
import pandas as pd
import re
from dateutil.relativedelta import relativedelta

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc


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
    

class PredictiveAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # initialize other variables
        self.X_name = []
        self.y_name = ''
        self.ma_opts = []
        self.fp_opts = []
        self.sc_opts = []
        self.datasets = {}

        # initialize hyperparameters
        self.eta_ = 0
        self.alpha_ = 0
        self.lambda_ = 0
        self.iter_ = 0
        
        # set variables for model evaluation    
        self.colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        self.titles = {
            'LinR': 'Linear Regression', 'LogR': 'Logistic Regression',
            'ACC': 'Accuracy', 'PRE': 'Precision', 'REC': 'Recall', 'F1': 'F1 score', 'AUC': 'Area Under ROC Curve'}
        
        self.results = {model: {} for model in ['LinR', 'LogR', 'CART']}  # theta, y_hat, sigma for each scope and dataset
        self.futures = {model: {} for model in ['LinR', 'LogR', 'CART']}  # predicted values for each 'fp'
        self.be_tests = {model: {} for model in ['LinR', 'LogR', 'CART']}  # backward elimination test results
        self.perf_df = {'LinR': pd.DataFrame(data=[], columns=['SC', 'MA', 'FP', 'RMSE', 'SE', 'R2', 'Adj-R2']),
                        'LogR': pd.DataFrame(data=[], columns=['SC', 'MA', 'FP', 'ACC', 'PRE', 'REC', 'F1', 'AUC']),
                        'CART': pd.DataFrame(data=[], columns=['SC', 'MA', 'FP', 'RMSE', 'SE', 'R2', 'Adj-R2'])}

        # initialize class
        self.poly = None
        self.scaler = None

        # initialize figures
        self.compere_perf_fig = None  # evaluation metrics
        self.be_test_fig = None  # backward elimination (scopes)
        self.coef_dev_fig = None  # development of coefficient over time 
        
        # css style for sub title
        self.SUB_CSS = 'style="font-size: 12.5px; color: lightgrey;"'

    def create_data(self, X_n: list, y_n: str, ma: list, fp: list, poly_d: int = 1):
        """
        Create the datasets -> self.datasets: dict

        Parameters:
        - "X_n": input feature names
        - "y_n": target value name
        - "ma": moving average options for a target value
        - "fp": options of future prediction (month basis);
                 how many months of target values are predictd based on current data.
        - "poly_degree": degree of the polynomials
        """
        # update variables
        self.y_name = y_n
        self.ma_opts = ma
        self.fp_opts = fp
        

        # start creating data
        y = np.array(self.df[[y_n]])
        for ma_i in self.ma_opts:
            # apply moving averages (ma is 0 or 1 -> no moving averages)
            if ma_i < 2:
                y_ma_i = y
                ma_i = 1
            else:
                #X_ma_i = np.array([np.mean(X[i-ma_i:i], axis=0) for i in range(ma_i, len(X))])
                y_ma_i = np.array([np.mean(y[i-ma_i:i], axis=0) for i in range(ma_i, len(y))])
            
            # apply polynomial
            self.poly = PolynomialFeatures(degree=poly_d, include_bias=True)
            X_poly = self.poly.fit_transform(self.df[X_n])
            # store all X names after polynomial transform
            self.X_name = self.poly.get_feature_names_out()
            
            # standardization
            self.scaler = Standardization()
            adjust = 0 if ma_i < 2 else ma_i 
            X_ss = self.scaler.fit_transform(120, X_poly[adjust:, 1:])
            # add bias term
            X_ss_b = np.c_[np.ones(X_ss.shape[0]), X_ss]

            # apply data shift
            for fp_i in self.fp_opts:
                self.datasets.update({
                    f"{ma_i}MA_{fp_i}FP": {'X': X_ss_b, 'y': y_ma_i[fp_i:], 'y_cat': np.where(y_ma_i[fp_i:]>0, 1, 0)}
                })


    def model_learning(self, scopes: list, model: str = '', eta_: float = 0.01, alpha_: float = 1, lambda_: float = 0.5, iter_: int = 100, th_=0.1):
        """
        Define and return three types of figures
        - "self.compere_perf_fig": comparing performance with respect to evaluation metrics.
        - "self.be_test_fig": result of the backward elimination with respect to scopes.
        - "self.coef_dev_fig": development of thetas(coefficisnts) over time.
        
        Parameters:
        - "scopes": how much previous data should be considered to update the parameters next step.
        - "model": either one of ['LinR', 'LogR', 'CART'].
        - "eta_": learning rate of each gradient descent.
        - "alpha_": degree of how strong the regularizations are.
        - "lambda_": balancer between l2 and l1 norm.
        - "iter_": maximum iteration of parameter updates at each step
        - "th_": the threshold for early stopping; applied in gradient of each parameter.
        """
        # define variable
        self.sc_opts = scopes

        # set hyperparameters globally
        self.eta_ = eta_
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.iter_ = iter_
        self.th_ = th_
 
        # set spaces (initialize)
        self.results = {model: {f'{i}FP': {} for i in self.fp_opts} for model in ['LinR', 'LogR', 'CART']}
        self.be_tests[model] = {sc: [] for sc in scopes}
        self.futures[model] = {f'{k1}FP': {f'{k2}MA': np.zeros((len(scopes), k1)) for k2 in self.ma_opts} for k1 in self.fp_opts}

        idx = 0
        # each scope
        for i, sc in enumerate(scopes):
            # each dataset
            for j, d_key in enumerate(self.datasets.keys()):
                # acquire data
                data = self.datasets[d_key]
                if model == 'LinR':
                    theta, y_hat, error, future = self.linear_reg(X=data['X'], y=data['y'], t=120, sc=sc)

                elif model == 'LogR':
                    theta, y_hat, error, future = self.logistic_reg(X=data['X'], y=data['y_cat'], t=120, sc=sc)

                elif model == 'CART':
                    theta, y_hat, error, future = None, None, None, None

                else:
                    raise TypeError('Choose one of following model names: "LinR", "LogR", and "CART".')
                
                # get moving average and future performance values from dataset keys
                ma, fp = d_key.split('_')
                # store all data
                self.results[model][fp].update({f"{ma}_{sc}SC": {'theta': theta, 'y_hat': y_hat, 'error': error}})
                # store future values
                self.futures[model][fp][ma][i] = future

                # get test result of backward elimination
                y = data['y_cat'] if model == 'LogR' else data['y']
                be_test_df = self.evaluation(model, data['X'], y, 120, theta, y_hat, error)
                # retrienve only performance without any changes in each coefficient
                ma_int = int(re.findall(r'\d+', ma)[0])
                self.perf_df[model].loc[idx] = [sc, ma_int, fp] + list(be_test_df.iloc[0])
                # store its result as NDArray
                self.be_tests[model][sc].append(np.array(be_test_df))

                # increment idx
                idx += 1
        
        self.compere_perf_fig = self.compare_perf(model)
        self.be_test_sc_fig = self.backward_elimination(model)
        self.coef_dev_fig = self.coefs_develop(model)

        return self.compere_perf_fig, self.be_test_sc_fig, self.coef_dev_fig


    def detail_perf(self, model: str, ma: int, fp: int, sc: int):
        """
        Return two plotly chart:
        - Line chart: Comparing the predicted and actual value.
        - Mix chart:  Distribution of the predicted errors by histogram and scatter plots.

        Return: plotly.graph_objs._figure.Figure

        """
        # get the results of model
        result = self.results[model]

        # get model and data
        d_name = f'{ma}MA_{fp}FP'
        data = self.datasets[d_name]
        _, y_hat, error = tuple(result[f'{fp}FP'][f'{ma}MA_{sc}SC'].values())
        
        # number of observations
        num_obs = len(y_hat)
        date = self.df['Date'][-num_obs:].values
        actual = data['y'][-num_obs:].flatten()
        predict = y_hat.flatten()

        # Figure 01 - Line Chart
        fig1 = go.Figure()
        x_date = pd.to_datetime(date)
        fig1.add_trace(go.Scatter(x=x_date, y=actual, mode='lines', name='Actual'))
        fig1.add_trace(go.Scatter(x=x_date, y=predict, mode='lines', name='Predict',
                                line=dict(color='lightgreen')))

        # future values
        y, m = x_date[-1].year, x_date[-1].month - 1
        months = [(i % 12) + 1 for i in range(m, m+fp+1)]
        add_on = []
        for i, month in enumerate(months):
            # update the year
            if i != 1 and month == 1:
                y += 1
            add_on.append(str(pd.Timestamp(y, month, 1) + pd.offsets.MonthEnd(1)).split()[0])
        # get the index of a scope
        sc_idx = self.sc_opts.index(sc)
        # get the future values
        future_vals = self.futures[model][f'{fp}FP'][f'{ma}MA'][sc_idx]
        fig1.add_trace(go.Scatter(
            x=pd.to_datetime(add_on), y=np.concatenate((y_hat[-1], future_vals)), 
            mode='lines', name='Future', line=dict(color='lightgreen', dash='dot')
            ))
        
        # layout
        add_title = '' if fp == 0 else f' in {fp}-Month '
        main_title = 'Prediction of %YoY S&P500' + add_title + 'Based on Current Data'
        sub_title = f'<br><span {self.SUB_CSS}> --Model predicts {ma}-month moving averaged target prices</span>'
        fig1.update_layout(
            height=400, width=700, template='plotly_dark', hovermode="x unified", title=main_title+sub_title, 
            legend=dict(orientation="h", yanchor="bottom", y=0.05, xanchor="left", x=0.0),
            margin=go.layout.Margin(l=80, r=40, b=40, t=80))

        # Figure 02 - Dostribution
        fig2 = go.Figure()
        # Scatter
        fig2.add_trace(
            go.Scatter(x=[i / 3 for i in range(len(error))], y=error.flatten(), mode='markers', 
                    marker=dict(color='red', opacity=0.5), name='Error',
                    hoverinfo='name+y+text',
                    text=["Date: " + d.strftime("%b %Y") for d in fig1.data[0]['x']],
                    ),
        )
        # histogram
        fig2.add_trace(
            go.Histogram(y=error.flatten(), nbinsy=20, marker=dict(color='grey', opacity=0.5), 
                        histnorm='', name='Distribution'),
        )
        # lauput
        main = "Distribution of the Predicted Errors"
        sub = f"<br><span {self.SUB_CSS}> -- Histogram shows the frequency of each predicted error.</span>"
        fig2.update_layout(
            height=400, width=700, template='plotly_dark', hovermode="x unified",
            title=main+sub, xaxis_title='Time Ranges', yaxis_title='YoY Growth (%)',
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.95),
            margin=go.layout.Margin(l=80, r=40, b=40, t=80))

        # Filter years that are divisible by 5 and month is January
        str_dates = x_date[(x_date.year % 5 == 0) & (x_date.month == 1)].strftime('%Y-%m-%d')
        # get the inde numbers 
        idx_dates = np.where(np.isin(date, str_dates))[0].tolist()
        fig2.update_xaxes(tickmode='array', tickvals=[i/3 for i in idx_dates], ticktext=[i[:4] for i in str_dates])

        return fig1, fig2


    def linear_reg(self, X, y, t, sc):
        """
        Return the following four matrix (dtype: np.array)
        - "thetas"  -> parameters (intercept + coefficients) at each step
        - "y_hats" -> predicted values at each step
        - "errors"  -> prediction errors (actual - predicted values); SSE
        - "future" -> Predicted values based on the rest of features

        Parameters:
        - "X": np.array -> independent variables
        - "y": np.array -> target variables
        - "t": int -> number of data that were used for the initial parameter creation.
        - "sc": int -> scope of the latest data for parameter updates


        Brief Steps:
        - Initialize matries for thetas, y_hats, errors.
        - Apply a given number of data ("t") to the mutiple linear regression (normal equation).
        - Define the initial parameters from the trained model.
        - At each step (total steps are len(y) - t):
            - Get a single pair of unfamilar data; both X and y.
            - Predict the target ("y_hats") based on the latest parameters("theta[i]").
            - Calculate the difference between actual and predicted values; "error[i]".
            - Update parameters for the next step ("theta[i+1]").
        """

        # define all matrix to be returned
        k = len(self.X_name) # num of features (bias inclusive)
        thetas = np.zeros((len(y[t:])+1, k))
        y_hats = np.zeros((len(y[t:]), 1))
        errors = np.zeros((len(y[t:]), 1))
        
        # initial training
        init_X, init_y = X[0:t], y[0:t]
        # obtain the initial parameters based on normal equation form
        thetas[0] = np.linalg.inv(init_X.T.dot(init_X)).dot(init_X.T).dot(init_y).reshape(1,-1)

        # define elastic net gradient
        def gradient(X, y, theta, n=sc, a_=self.alpha_, l_=self.lambda_):
            # gradient
            grad = -2/n * (np.dot(X.T, y - np.dot(X, theta.T))).T
            d_l1 = l_ * a_ * np.sign(theta)
            d_l2 = (1 - l_) * a_ * theta
            return grad + d_l1 + d_l2
        
        # incremental learnings
        for i, idx in enumerate(range(t, len(y), 1)):
            # get new data (X_i includes bias)
            X_i, y_i = X[idx] , y[idx]
            # predicted variable
            y_hats[i] = np.dot(X_i, thetas[i])
            # predicted error
            errors[i] =  (y_hats[i] - y_i)
            # define subsets of X and y
            X_sub, y_sub = X[idx-sc+1:idx+1], y[idx-sc+1:idx+1]
            # get the current theta
            theta_ = thetas[i:i+1]
            # iterations
            for _ in range(self.iter_):
                # get the partial derivative
                grad = gradient(X_sub, y_sub, theta_)
                # define early stopping
                if np.all(np.abs(grad) < self.th_):
                    break
                # update the theta
                theta_ -= self.eta_* grad
            
            # assign finialized theta
            thetas[i+1] = theta_

        # calculate the future values
        future = np.sum(np.multiply(X[len(y):], thetas[len(y)-t:]), axis=1)

        return thetas, y_hats, errors, future


    def logistic_reg(self, X, y, t, sc):
        """
        Return the following five matrix (dtype: np.array)
        - "thetas"  -> parameters (intercept + coefficients) at each step
        - dictionary: predicted y values:
            - "cat" key: "y_preds_c" -> predicted labels at each step
            - "proba" ley: "y_preds_p" -> predicted probability at each step (nagetige & positive class)
        - "errors"  -> prediction errors (actual - predicted values); SSE
        - dictionary: future y values 
            - "cat": predicted labels at each step
            - "proba": probability of the positive class label

        Parameters:
        - "X": np.array -> independent variables
        - "y": np.array -> target variables (shouold be categorical)
        - "t": int -> number of data that were used for the initial parameter creation.
        - "sc": int -> scope of the latest data for parameter updates.

        Brief Steps:
        - Initialize all matries to store ithe ncrementally updated values.
        - Apply a given number of data ("t") to the mutiple linear regression (normal equation).
        - Define the initial parameters from the trained model.
        - At each step (total steps are len(y) - t):
            - Get a single pair of unfamilar data; both X and y.
            - Predict the target ("y_hats") based on the latest parameters("theta[i]").
            - Calculate the difference between actual and predicted values; "error[i]".
            - Update parameters for the next step ("theta[i+1]").
        """
        # define sigmoid function
        def sigmoid(h):
            return 1 / (1 + np.exp(-h))

        # calculate the gradient vector
        def gradient(X, y, theta, w, alpha_, lambda_):
            m = len(y)
            preds = sigmoid(np.dot(X, theta.reshape(-1,1)))
            grad = -np.dot(X.T, np.multiply(y - preds, w)) / m
            l1 = lambda_ * np.sign(theta)
            l2 = (1 - lambda_) * theta

            return grad.T + alpha_ * (l1 + l2)
        
        # initialize matrics to store values at each step
        thetas = np.zeros((len(y[t:])+1, 6))
        y_hats = np.zeros((len(y[120:]), 1))
        errors = np.zeros((len(y[t:]), 1))
        # initial training 
        logit = LogisticRegression(fit_intercept=False, class_weight='balanced')
        logit.fit(X[:t], y[:t].flatten())
        # set initial parameters
        thetas[0] = logit.coef_
        # define class weights
        weights = {u: len(X) / (2*np.bincount(y.flatten())[u]) for u in np.unique(y)}
        # define the vectorized function to converting class labels to weights
        vfunc_weights = np.vectorize(lambda x: weights[x])

        # start incremental learning
        for idx, i in enumerate(range(120, len(y), 1)):
            # apply sigmoid function
            y_hats[idx] = sigmoid(np.dot(X[i:i+1], thetas[idx]))
            # actual value
            y_act = y[i][0]
            # logistic loss
            y_proba = {0: 1-y_hats[idx], 1: y_hats[idx]}
            errors[idx] = -1 * weights[y_act] * np.log(y_proba[y_act])

            # subset pf X and y
            X_sub, y_sub = X[i+1-sc: i+1], y[i+1-sc: i+1]
            # update parameter
            theta_epoch = thetas[idx]
            for i in range(self.iter_):
                # get the partial derivative
                grad = gradient(X_sub, y_sub, theta_epoch, vfunc_weights(y_sub), self.alpha_, self.lambda_)
                # define early stopping
                if np.all(np.abs(grad) < self.th_):
                    break
                # update the theta
                theta_epoch -= self.eta_* grad.flatten()

            thetas[idx+1] = theta_epoch

        # predict the future labels
        future = sigmoid(np.dot(X[len(y):], thetas[-1]))

        return thetas, y_hats, errors, future


    def evaluation(self, model, X, y, t, thetas, y_hats, errors):
        """
        Return the data frame; the following measureas:
        (1): Numerical Target Value
            - Root Mean Square Error (rmse)
            - Standatd Error of Estimate (se)
            - Coefficient of Determination (r2)
            - Adjusted Coefficient of Determination (adj_r2)

        (2): Categorical Target Value
            - Accuracy (acc)
            - Precsion (pre)
            - Recall (rec)
            - F1 Score (f1)
            - Area Under the ROC Curve (auc)

        Parameters:
        - "model": either one of ['LinR', 'LogR', 'CART']
        - "t": number of months that were used for the initial training data.
        - "X": independent variables
        - "y": the actual target value (whole period)
        - "theta": all parameters (intercept & coefficients) at each step
        - "y_hats": the predicted target value at each step
        - "error": difference between actual and predicted values at each step

        Requirement: len(y[t:]) == len(y_hats)
        """
        #  number of observations and features (excluding bias)
        n, k = len(y_hats), len(thetas[0]) - 1
        # define feature matrix and target based on "t"
        X_, y_ = X[t:n+t], y[t:]
        # assign the data frame index and columns
        rows = ['original'] + [f'theta{i+1}=0' for i in range(k)]

        # define measures based on model
        if model == 'LinR':
            # names of the measure
            cols = ['rmse', 'se', 'r2', 'adj_r2']
            # measures matrix
            mm = np.zeros((k+1, len(cols)))
            # sum of square total
            sst = np.sum((y_ - np.mean(y_))**2)
            #  number of coefficients
            for i in range(k+1):
                # simply applying the given error
                if i == 0:
                    error_ = errors
                # conduct the backward elimination
                else:
                    # copy the parameters matrix
                    theta_ = thetas[:n].copy()
                    # change a particular coefficient to 0 arbitrarily.
                    theta_[:, i] = 0
                    # based on revised parameters, get the predicted value
                    y_hats_ = np.sum(X_ * theta_, axis=1).reshape(-1, 1)
                    # predicted error
                    error_ = y_ - y_hats_    
            
                # Calculate Measures (rmse, se, r2, adj_r2, in order)
                sse = np.sum(error_**2)
                mm[i, 0] = np.sqrt((error_**2).mean()) 
                mm[i, 1] = np.sqrt(sse / (n - k - 1))
                mm[i, 2] = 1 - sse/sst
                mm[i, 3] = 1 - (sse/(n - k -1))/(sst/(n-1))

            return pd.DataFrame(data=mm, index=rows, columns=cols)

        elif model == 'LogR':
            # define functions to generate classification measures from confusion matrix
            def measures_from_cm(cm):
                tn, fp, fn, tp = cm.ravel()
                #calculate the metrics
                acc = (tp + tn) / np.sum(cm)  # accuracy
                prec = tp / (tp + fp)  # precision
                rec = tp / (tp + fn)  # recall
                f1 = 2 * prec * rec / (prec + rec)  # f1 score
                fpr = fp / (fp + tn)  # false positive rate
                tpr = tp / (tp + fn)  # true positive rate
                # points for roc
                roc_points = [(0, 0), (fpr, tpr), (1, 1)]
                auc_ = auc([p[0] for p in roc_points], [p[1] for p in roc_points])

                return [acc, prec, rec, f1, auc_]
            
            # names of the measures
            cols = ['acc', 'pre', 'rec', 'f1', 'auc']
            # measures matrix
            mm = np.zeros((k+1, len(cols)))
            #  number of coefficients
            for i in range(k+1):
                # simply applying the predicted label
                if i == 0:
                    y_hats_ = np.where(y_hats >= 0.5, 1, 0)
                # conduct the backward elimination
                else:
                    # copy the parameters matrix
                    theta_ = thetas[:n].copy()
                    # change a particular coefficient to 0 arbitrarily.
                    theta_[:, i] = 0
                    # based on revised parameters, get the predicted value
                    hx = np.sum(X_ * theta_, axis=1).reshape(-1, 1)
                    y_hats_ = np.where(1 / (1 + np.exp(-hx)) >= 0.5, 1, 0)
            
                # confusion matrix
                cm = confusion_matrix(y_, y_hats_)
                # assign each measure              
                measures = measures_from_cm(cm)
                for j, val in enumerate(measures):
                    mm[i, j] = val

            return pd.DataFrame(data=mm, index=rows, columns=cols)

        else:
            raise(TypeError("Model should be either one of ['LinR', 'LogR', 'CART']"))


    def compare_perf(self, model: str):
        # deine measures
        measures = list(self.perf_df['LinR'].columns)[3:]
        # modity perf_df
        perf_df = self.perf_df['LinR'].drop('SC', axis=1).groupby(['FP', 'MA']).mean()
        #perf_df.columns = ['_'.join(col) for col in perf_df.columns]
        perf_df = perf_df.reset_index()
        perf_df['MA'] = perf_df['MA'].astype(str)

        figs = []
        # traversing all measures
        for ms in measures:
            # define fifure
            fig = px.scatter(perf_df, x='FP', y=ms, color='MA')

            fig.update_xaxes(title_text='Moving Averages', title_font={'color':'lightgrey'})
            fig.update_layout(
                height=300, width=300, template='plotly_dark',
                title=dict(text=self.titles[ms] if self.titles.get(ms) else ms,  xanchor="center", x=0.50),
                showlegend=False, yaxis_title='', 
                margin=go.layout.Margin(t=50, l=30, r=30, b=50)
                ) 

            figs.append(fig)
            
        return figs


    def backward_elimination(self, model: str):
        """
        Evaluate the backward elimination for all models with the following focuses
            - RMSE and adjusted R2 for regression
            - Accuracy and f1 score and for classification
        Visualize the scatter plots to show the difference from original result

        Parameter:
        - 'model': either one of models: 'LinR', 'LogR', 'CART'
        """
        be_test = self.be_tests[model]
        # number of features; excluding bias term
        n = len(self.X_name) - 1
        # define dictionary for two measures
        m1_d = {'sc': [], 'theta': [], 'diff': []}  # rmse (reg), acc (cls)
        m2_d = {'sc': [], 'theta': [], 'diff': []}  # adj-r2 (reg), f1 (cls)
        # define location of focusing measures
        if model == 'LogR':
            idx1 = 0
            idx2 = -2
            names = ['Acc', 'F1']
        else:
            idx1 = 0
            idx2 = -1
            names = ['RMSE', 'Adj-R2']

        for key in be_test:
            for matrix in be_test[key]:
                # learning method
                m1_d['sc'] += [key] * (n)
                m2_d['sc'] += [key] * (n)
                # add theta name
                m1_d['theta'] += list(self.X_name[1:])
                m2_d['theta'] += list(self.X_name[1:])
                # first column is RMSE and last one is adjusted R2
                diff = matrix[1:] - matrix[0]
                # add error
                m1_d['diff'] += list(diff[:, idx1])
                m2_d['diff'] += list(diff[:, idx2])
                

        # plot data points
        m1_df = pd.DataFrame(m1_d)
        m2_df = pd.DataFrame(m2_d)
        fig1 = px.strip(data_frame=m1_df, x='theta', y='diff', color='sc')
        fig2 = px.strip(data_frame=m2_df, x='theta', y='diff', color='sc')

        # set figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=names)
        # Update y position of subplot titles
        fig.layout.annotations[0].update(y=0.95, font=dict(size=11, color='grey'))
        fig.layout.annotations[1].update(y=0.95, font=dict(size=11, color='grey'))

        # add the trace objects to the subplots
        for fig_loc in range(len(be_test.keys())):
            # add figure data one by one
            fig.add_trace(fig1.data[fig_loc], row=1, col=1)
            # remove duplicated legend
            fig2.data[fig_loc].showlegend = False
            fig.add_trace(fig2.data[fig_loc], row=1, col=2)

        fig.update_traces(marker=dict(opacity=0.5, size=5))

        # add layout
        main = f"Observe Backward Elimination in All Models With Repect to Scopes" 
        sub = f"<br><span {self.SUB_CSS}> -- How meansures are changed by removing the impact of each coefficient</span>"
        fig.update_layout(height=400, width=800, template='plotly_dark', 
                        title_text=main + sub, yaxis_title="Difference",
                        legend=dict(title_text='Scopes:', orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                        margin=go.layout.Margin(t=80, b=60, l=80, r=40))
    
        return fig
    
    
    def coefs_develop(self, model: str):
        """
        Generate the development of each coefficient impacts over time on average.

        Parameters:
        - "model": either one of defined model names; "LinR" and "LogR"

        """
        # get the minimum theta dimentions (ecluding bias term)
        max_fp, max_ma, first_sc = max(self.fp_opts), max(self.ma_opts), self.sc_opts[0]
        thetas_dim = self.results[model][f'{max_fp}FP'][f'{max_ma}MA_{first_sc}SC']['theta'][:, 1:].shape
        # get the last date of data frame
        last_date = pd.to_datetime(self.df['Date'].iloc[-1])
        # get the next month last date
        next_month_last = last_date + relativedelta(months=1)
        # define start date
        start_date = next_month_last - relativedelta(months=thetas_dim[0]-1)
        # define date ranges
        date_ranges = np.array(pd.date_range(start_date, next_month_last, freq='M'))
        # define data frame
        thetas_df = pd.DataFrame({"Date": date_ranges})

        # initialize thetas data
        thetas_data = np.zeros(thetas_dim)
        denominator = 0
        # get all thetas from multiple models
        for k1 in self.results[model].keys():
            for k2 in self.results[model][k1].keys():
                thetas_data += self.results[model][k1][k2]['theta'][-thetas_dim[0]:, 1:]
                denominator += 1

        # calculate mean
        thetas_mean = thetas_data / denominator
        # define column names
        cols = self.X_name[1:]
        # adding thetas_data
        thetas_df = pd.concat([thetas_df, pd.DataFrame(thetas_mean, columns=cols)], axis=1)
        
        # create visualizations
        fig = px.line(thetas_df, 'Date', cols)
        for trace in fig.data:
            trace.hovertemplate = f'%{{y}}'

        fig.update_layout(
            title='Impact of Economic Indicators on S&P500 Over Time',
            legend=dict(title_text='Indicators', orientation='h', font_color='lightgray',
                        x=-0.05, y=1.05, xanchor='left', yanchor='top'),
            template='plotly_dark', hovermode="x unified",
            yaxis=dict(title_text='Strengths'),
            height=400, width=800, margin=go.layout.Margin(t=60, b=50, l=50, r=30),
            )
        
        return fig
    
