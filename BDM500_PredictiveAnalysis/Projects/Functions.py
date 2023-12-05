import numpy as np
from numpy.typing import NDArray
import pandas as pd
import re
from dateutil.relativedelta import relativedelta
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from copy import deepcopy
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

    def fit_transform(self, X: NDArray, t: int):
        """
        Standarize X features by mean and std of the time periods of "t", then return the standardized features.
        
        Parameters:
        "t": observing time ranges or number of data; a window of data.
        "X": the matrix of independent variables.
        """
        # store all given parameters
        self.X = X
        self.t = t
        self.X_ss = np.zeros(X.shape)

        # initial standardization
        self.mu, self.sd = np.mean(X[0:t], axis=0), np.std(X[0:t], axis=0)
        self.X_ss[0:t] = (X[0:t] - self.mu) / self.sd

        # standarize new feature one by one
        for i, idx in enumerate(range(t, len(X))):
            # assign a series of newly standardized features
            self.X_ss[idx] = self.update(x_del=X[i], x_new=X[idx], update=True)

        return self.X_ss
        

    def transform(self, x: NDArray, update: bool = True):
        """
        Return value is the same as self.update().
        A given series of features ("x") is added to the stored feature matrix.

        This function is invoked everytime when a new series of new features.
        Make sure the consistency of the time series data. 

        Parameter:
        - "x": a series of newly updated features and has never been add in self.X 
        - "update": if True, updating the newly added vector of independent variables
        """
        # a series of deleted features 
        x_del = self.X[-self.t]
        # assign standardized features
        new_features = self.update(x_del=x_del, x_new=x, update=update)
        # update X
        if update:
            self.X = np.append(self.X, x)
            self.X_ss = np.append(self.X_ss, new_features)

        return new_features


    def update(self, x_del: NDArray, x_new: NDArray, update: bool):
        """
        Return a series of standardized features one by one based on the mean and std of the most recent "self.t" number of data.
        In other words, a series of new features are standardized by the moving average and std over the "self.t" periods.

        Paratermers:
        - "x_del": a series of unscoped features.
        - "x_new": a series of the most recent features.
        - "update": wheteth the current mu and std is updated or not
        """
        # assign current mean and std as old ones
        mu_old, sd_old = self.mu, self.sd
        mu = mu_old + (x_new - x_del) / self.t
        sd = np.sqrt(((self.t-1)*sd_old**2 - (x_del - mu_old)**2 + (x_new - mu)**2) / (self.t-1))

        # update if True
        if update:
            self.mu_old, self.sd_old = mu_old, sd_old 
            self.mu, self.sd = mu, sd

        # assign standardized features
        new_features = (x_new - mu) / sd

        return new_features
    

class PredictiveAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # initialize other variables
        self.new_df = pd.DataFrame()
        self.X_name = []
        self.y_name = ''
        self.X = np.array([])
        self.y_num = np.array([])
        self.y_cat = np.array([])
        self.ma_opts = []
        self.fp_opts = []
        self.sc_opts = []
        self.init_train = 0  # number of initial training data
        self.trees = {}  # regression tree model and each step; keys MA and FP

        # initialize hyperparameters
        self.eta_ = 0
        self.alpha_ = 0
        self.lambda_ = 0
        self.iter_ = 0
        
        # set variables for model evaluation    
        self.colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        self.titles = {
            'LinR': 'Linear Regression', 'LogR': 'Logistic Regression',
            'ACC': 'Accuracy', 'PRE': 'Precision', 'REC': 'Recall', 'F1': 'F1 score', 'AUC': 'Area Under ROC'}
        
        self.results = {model: {} for model in ['LinR', 'LogR', 'CART']}  # theta, y_hat, sigma for each scope and dataset
        self.pred_df = {model: pd.DataFrame() for model in ['LinR', 'LogR', 'CART']}  # pd.DataFrame of all predicted values
        self.futures = {model: {} for model in ['LinR', 'LogR', 'CART']}  # predicted values for each 'fp'
        self.be_tests = {model: {} for model in ['LinR', 'LogR', 'CART']}  # backward elimination test results
        self.perf_df = {'LinR': pd.DataFrame(data=[], columns=['MA', 'FP', 'SC', 'RMSE', 'SE', 'R2', 'Adj-R2']),
                        'LogR': pd.DataFrame(data=[], columns=['MA', 'FP', 'SC', 'ACC', 'PRE', 'REC', 'F1', 'AUC']),
                        'CART': pd.DataFrame(data=[], columns=['MA', 'FP', 'RMSE', 'SE', 'R2', 'Adj-R2'])}

        # initialize class
        self.poly = None
        self.scaler = None

        # css style for sub title
        self.SUB_CSS = 'style="font-size: 12.5px; color: lightgrey;"'

    def create_data(self, X_n: list, y_n: str, ma: list[int], fp: list[int], init_train: int = 120, poly_d: int = 1):
        """
        Create the datasets (defined variables are shown below)
        - "self.X_name": list -> each feature name (including bias)
        - "self.X": NDArray -> applying polynomial and Standarization
        - "self.y_name": list -> each column's y name
        - "self.y_num" & "self.y_cat": NDArray -> applying moving averages 

        Parameters:
        - "X_n": input feature names
        - "y_n": target value name
        - "ma": moving average options for a target value
        - "fp": options of future prediction (month basis);
                 how many months of target values are predictd based on current data.
        - "init_train": range of data for initial training
        - "poly_degree": degree of the polynomials
        """
        # update variables
        self.ma_opts = ma
        self.fp_opts = fp
        self.init_train = init_train

        # X fetures matrix
        # apply polynomial
        self.poly = PolynomialFeatures(degree=poly_d, include_bias=True)
        X_poly = self.poly.fit_transform(self.df[X_n].iloc[12:])
        # store all X names after polynomial transform
        self.X_name = self.poly.get_feature_names_out()
        # standardization
        self.scaler = Standardization()
        X_ss = self.scaler.fit_transform(X_poly[:, 1:], self.init_train)
        # add bias term
        self.X = np.c_[np.ones(X_ss.shape[0]), X_ss]

        # y matrix (each column shows different types of y)
        self.y_name = [f'{y_n}_{ma}MA' for ma in self.ma_opts]
        # initialize y matrix
        self.y_num = np.full((self.X.shape[0], len(self.y_name)), np.nan)
        self.y_cat = np.full((self.X.shape[0], len(self.y_name)), np.nan)
        # original y
        origin_y = np.array(self.df[y_n])
        for c, ma in enumerate(self.ma_opts):
            # apply moving averages (ma is 0 or 1 -> no moving averages)
            if ma < 2:
                self.y_num[:, c] = origin_y[12:]
                self.y_cat[:, c] = np.where(self.y_num[:, c] > 0, 1, 0)
            else:
                self.y_num[:, c] = np.array([np.mean(origin_y[i-ma:i], axis=0) for i in range(ma, len(origin_y)+1)])[12-ma+1:]
                self.y_cat[:, c] = np.where(self.y_num[:, c] > 0, 1, 0)

        # create new dataframe
        new_df_info = {'Date': self.df['Date'].iloc[12:].reset_index(drop=True)}
        new_df_info.update({
            **{key: self.X[:, i+1] for i, key in enumerate(self.X_name[1:])},
            **{f'{key}_num': self.y_num[:, i] for i, key in enumerate(self.y_name)},
            **{f'{key}_cat': self.y_cat[:, i] for i, key in enumerate(self.y_name)}
        }) # type: ignore

        self.new_df = pd.DataFrame(new_df_info)
        return self.new_df


    def model_learning(self, model: str, scopes: list = [1,3,6,9,12], X_use: list = [], 
                       eta_: float = 0.01, alpha_: float = 0.1, lambda_: float = 0.5, iter_: int = 100,
                       max_death_: int = 5):
        """
        Define and return three types of figures
        - "fig1": comparing performance with respect to evaluation metrics.
        - "fig2": result of the backward elimination with respect to scopes.
        - "fig3": impacts of each economic indicator on target over time.
        
        Parameters:
        - "scopes": how much previous data should be considered to update the parameters next step.
        - "model": either one of ['LinR', 'LogR', 'CART'].
        - "X_use": the list of feature names that are used for the model training.
        (for Linear and Logistic Regression models)
        - "eta_": learning rate of each gradient descent.
        - "alpha_": degree of how strong the regularizations are.
        - "lambda_": balancer between l2 and l1 norm.
        - "iter_": maximum iteration of parameter updates at each step
        (For classification and regression tree model)
        - "max_death_": 

        """
        # define variable
        self.sc_opts = scopes

        # update pred_df
        self.pred_df[model] = pd.DataFrame()
        self.pred_df[model]['Date'] = self.new_df['Date'].iloc[self.init_train:].reset_index(drop=True)
        # numerical or categorical target
        type_ = 'cat' if model == 'LogR' else 'num'
        # define df for concatenate
        concat_df = pd.DataFrame({
            f'Act_{y_n.split("_")[1]}': self.new_df[f'{y_n}_{type_}'].iloc[self.init_train:].reset_index(drop=True) for y_n in self.y_name
        })
        # concatenating them
        self.pred_df[model] = pd.concat([self.pred_df[model], concat_df], axis=1)
        
        # adding new rows with NaN for future prediction
        start_date = pd.to_datetime(self.pred_df[model]['Date'].iloc[-1]) + pd.DateOffset(months=1)
        new_dates = pd.date_range(start=start_date, periods=max(self.fp_opts), freq='M')
        new_rows = pd.DataFrame({
            'Date': new_dates.strftime('%Y-%m-%d'),
        })
        self.pred_df[model] = pd.concat([self.pred_df[model], new_rows], ignore_index=True)

        # define X feature index that will be used for model training
        if not X_use:
            # do not use bias term for CART model
            adj_bias = 1 if model == 'CART' else 0
            self.X_use_idx = [i for i in range(adj_bias, len(self.X_name))]
        else:
            if model == 'CART':
                # always removing bias term
                self.X_use_idx = list(np.where(np.in1d(self.X_name, X_use))[0])
            else:
                self.X_use_idx = [0] + list(np.where(np.in1d(self.X_name, X_use))[0])

        # set hyperparameters
        self.eta_ = eta_
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.iter_ = iter_

        # set spaces (initialize)
        self.results = {model: {f'{i}FP': {} for i in self.fp_opts} for model in ['LinR', 'LogR', 'CART']}
        self.be_tests[model] = {sc: [] for sc in scopes}
        self.futures[model] = {f'{k1}FP': {f'{k2}MA': np.zeros((len(scopes), k1)) for k2 in self.ma_opts} for k1 in self.fp_opts}

        idx = 0
        # each moving average
        for i, ma in enumerate(self.ma_opts):
            # each future prediction
            for fp in self.fp_opts:
                # each scope
                for j, sc in enumerate(self.sc_opts):
                    # linear regression
                    if model == 'LinR':
                        theta, y_hat, error, future = self.linear_reg(X=self.X[:, self.X_use_idx], y=self.y_num[:, [i]], t=self.init_train, fp=fp, sc=sc)
                    
                    elif model == 'LogR':
                        theta, y_hat, error, future = self.logistic_reg(X=self.X[:, self.X_use_idx], y=self.y_cat[:, [i]], t=self.init_train, fp=fp, sc=sc)

                    elif model == 'CART':
                        theta, y_hat, error, future, models = self.tree_reg(X=self.X[:, self.X_use_idx], y=self.y_num[:, [i]], t=self.init_train, fp=fp, max_death_=max_death_)
                        # store all tree diagram
                        self.trees[f'{ma}MA_{fp}FP'] = models

                    else:
                        raise TypeError('Choose one of following model names: "LinR", "LogR", and "CART".')
                    
                    # update predicted value and store all results
                    add_array = np.append(future, np.full((max(self.fp_opts)-fp), np.nan))
                    if model == 'CART':
                        self.pred_df[model][f'Pred_{ma}MA_{fp}FP'] = np.append(y_hat, add_array)
                        self.results[model][f'{fp}FP'].update({f"{ma}MA": {'f_impacts': theta, 'y_hat': y_hat, 'error': error}})
                    else:
                        self.pred_df[model][f'Pred_{ma}MA_{fp}FP_{sc}SC'] = np.append(y_hat, add_array)
                        self.results[model][f'{fp}FP'].update({f"{ma}MA_{sc}SC": {'theta': theta, 'y_hat': y_hat, 'error': error}})

                    # store future values
                    self.futures[model][f'{fp}FP'][f'{ma}MA'][j] = future.flatten()

                    # get test result of backward elimination for LinR and LogR or metrics for CART
                    y_vec = self.y_cat[:, [i]] if model == 'LogR' else self.y_num[:, [i]]
                    evaluation_df = self.evaluation(model, self.X[:, self.X_use_idx], y_vec, self.init_train, theta, y_hat, error, fp)
                    

                    if model == 'CART':
                        # retrienve only performance without any changes in each coefficient
                        self.perf_df[model].loc[idx] = [ma, fp] + list(evaluation_df.iloc[0])
                    else:
                        self.perf_df[model].loc[idx] = [ma, fp, sc] + list(evaluation_df.iloc[0])

                    # store its result as NDArray
                    self.be_tests[model][sc].append(np.array(evaluation_df))

                    # increment idx
                    idx += 1

                    # CART model does not take care of different scopes
                    if model == 'CART' and j == 0:
                        break

        # generate visualizations
        # comparing various metrics
        fig1 = self.compare_perf(model)
        if model == 'CART':
            fig2 = go.Figure()
        else:
            # backward elimination results
            fig2 = self.backward_elimination(model)

        # impacts of features on target over time
        fig3 = self.coefs_develop(model)

        return fig1, fig2, fig3


    def detail_perf(self, model: str, ma: int, fp: int|str = 'mean', sc: int|str = 'mean'):
        """
            Return two plotly chart:
            - Line chart: Comparing the predicted and actual value.
            - Mix chart:  Distribution of the predicted errors by histogram and scatter plots.

            Return: plotly.graph_objs._figure.Figure

        """
        # get pred_df
        pred_df = self.pred_df[model].set_index("Date")
        
        # (1): Line Charts
        fig1 = go.Figure()
        # common x-axis data
        x_date = pd.to_datetime(pred_df.index.values)
        # actual value
        actual = pred_df[f'Act_{ma}MA']
        fig1.add_trace(go.Scatter(x=x_date, y=actual, mode='lines', name='Actual'))

        # function to add bands (3 stds)
        def add_bands(fig, predict, cols):
            sigma_3 = (pred_df[cols].std(axis=1) * 3).values
            fig.add_traces([
                go.Scatter(
                    name='+-3 sigma',
                    x=x_date, y=predict - sigma_3, 
                    mode='lines', line=dict(width=0, color='rgba(255, 255, 255, 0)'),
                    legendgroup='bands', hovertemplate='Lower: %{y:.2f}<extra></extra>', showlegend=False
                ),
                go.Scatter(
                    name='+-3 sigma', 
                    x=x_date, y=predict + sigma_3,
                    mode='lines', line=dict(width=0, color='rgba(255, 255, 255, 0)'),
                    fillcolor='rgba(255, 255, 255, 0.2)', fill='tonexty', 
                    legendgroup='bands', hovertemplate='Upper: %{y:.2f}<extra></extra>', showlegend=True
                ),
            ])
            return fig

        # predictions
        draw_bands = True
        #  if both 'fp' and 'sc' are number
        if fp != 'mean' and sc != 'mean':
            filter_cols = f'Pred_{ma}MA_{fp}FP_{sc}SC'
            predict = pred_df[filter_cols]
            sub_title1 = f'<br><span {self.SUB_CSS}> --Adjusted parameters to predict a target price {fp} month(s) ahead.</span>'
            sub_title2 = f'<br><span {self.SUB_CSS}> --Focused on {sc} month(s) of data to update parameters at each step.</span>'
            draw_bands = False
        #  if only sc is 'mean
        elif fp != 'mean' and sc == 'mean':
            filter_cols = [c for c in pred_df.columns if c.startswith(f'Pred_{ma}MA_{fp}FP')]
            predict = pred_df[filter_cols].mean(axis=1)
            sub_title1 = f'<br><span {self.SUB_CSS}> --Adjusted parameters to predict a target price {fp} month(s) ahead.</span>'
            sub_title2 = f'<br><span {self.SUB_CSS}> --Aggregated different prediction results generated from all scopes.</span>'
            if model == 'CART':
                draw_bands = False
        #  if only fp is 'mean'
        elif fp == 'mean' and sc != 'mean':
            filter_cols = [c for c in pred_df.columns 
                        if c.startswith(f'Pred_{ma}MA') and c.endswith(f'_{sc}SC')]
            predict = pred_df[filter_cols].mean(axis=1)
            sub_title1 = f'<br><span {self.SUB_CSS}> --Aggregated multiple results generated from all defined future months ahead.</span>'
            sub_title2 = f'<br><span {self.SUB_CSS}> --Focused on the recent {sc} month(s) of data to update parameters at each step.</span>'
        #  else; both are 'mean'
        else:
            filter_cols = [c for c in pred_df.columns if c.startswith(f'Pred_{ma}MA')]
            predict = pred_df[filter_cols].mean(axis=1)
            sub_title1 = f'<br><span {self.SUB_CSS}> --Aggregated all results from different futures ahead & scopes.</span>'
            sub_title2 = f'<br><span {self.SUB_CSS}> --Bands showing three standard deviations based on all results. </span>'

        # drawing line
        fig1.add_trace(go.Scatter(x=x_date, y=predict, mode='lines', name='Predict',
                                line=dict(color='lightgreen')))
        # drawing bands
        if draw_bands:
            fig1 = add_bands(fig1, predict, filter_cols)

        # layout
        if ma == 1:
            add_title = '(Actual Prices)'
        else:
            add_title = f'({ma}-Month Moving Averaged Prices)'
        # adjustment
        adj_y = 0
        if model == "CART":
            sub_title2=''
            adj_y = 0.05

        main_title = 'Prediction of %YoY S&P500 ' + add_title
        fig1.update_layout(
            height=400, width=700, template='plotly_dark', hovermode="x unified", 
            title=dict(text=main_title+sub_title1+sub_title2, y=0.90-adj_y),
            legend=dict(orientation="h", yanchor="bottom", y=0.05, xanchor="left", x=0.0),
            margin=go.layout.Margin(l=80, r=40, b=40, t=100)
        )
        fig1.update_traces(
            selector=dict(name='3 sigma'),
            line=dict(color='rgba(255, 255, 255, 0.2)'), 
            overwrite=True 
        )


        # (2): Distribution Charts
        # calculate error
        error = (actual - predict).values
        # the highest frequency (max count)
        hist, _ = np.histogram(error[~np.isnan(error)], bins=10)
        max_freq= max(hist)
        # scale change(max_freq is assigned 70% of the final plots)
        total_obs = len(x_date)
        scale_chg = (total_obs*max_freq / (total_obs*0.7)) / total_obs
        # define figure
        fig2 = go.Figure() 
        # adding fig data
        fig2.add_traces([
            # distribution plots
            go.Histogram(
                y=error, nbinsy=20, marker=dict(color='grey', opacity=0.5), 
                histnorm='', name='Distribution'
            ),
            # error plots
            go.Scatter(
                x=[i * scale_chg for i in range(len(x_date))], y=error, mode='markers', 
                marker=dict(color='red', opacity=0.5), name='Error',
                hoverinfo='name+y+text', text=["Date: " + d.strftime("%b %Y") for d in fig1.data[0]['x']],
            ),
        ])
        
        # lauput
        main = "Distribution of the Predicted Errors"
        sub = f"<br><span {self.SUB_CSS}> -- Histogram shows the frequency of each predicted error.</span>"
        fig2.update_layout(
            height=400, width=700, template='plotly_dark', hovermode="x unified",
            title=main+sub, yaxis_title='YoY Growth (%)',
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.95),
            margin=go.layout.Margin(l=80, r=40, b=40, t=80))

        # Filter years that are divisible by 5 and month is January
        ticks_dates = x_date[(x_date.year % 5 == 0) & (x_date.month == 1)]

        # get the index numbers 
        idx_dates = np.where(np.isin(x_date, ticks_dates))[0].tolist()
        fig2.update_xaxes(tickmode='array', 
                        tickvals=[i*scale_chg for i in idx_dates], 
                        ticktext=[i[:4] for i in ticks_dates.strftime('%Y-%m-%d')])

        return fig1, fig2


    def linear_reg(self, X, y, t, fp, sc):
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
        - "fp": the performance of certain months of future
        - "sc": int -> scope of the latest data for parameter updates
        
        """
        # define all matrix to be returned
        thetas = np.full((X.shape[0]-t, X.shape[1]), np.nan)
        y_hats = np.full((X.shape[0]-t, 1), np.nan)
        errors = np.full((X.shape[0]-t, 1), np.nan)
        
        # initial training
        init_X, init_y = X[0:t], y[fp:t+fp]
        # obtain the initial parameters based on normal equation form
        thetas[:fp] = np.linalg.inv(init_X.T.dot(init_X)).dot(init_X.T).dot(init_y).reshape(1,-1)

        # define elastic net gradient
        def gradient(X, y, theta, n=sc, a_=self.alpha_, l_=self.lambda_):
            """
            X: (n, k)
            y: (n, k)
            theta: (k, 1)
            
            'n' is number of the recent observations (scopes)
            'k' is number of parameters (including bias term)
            """
            # gradient
            grad = -2/n * (np.dot(X.T, y - np.dot(X, theta)))
            d_l1 = l_ * a_ * np.sign(theta)
            d_l2 = (1 - l_) * a_ * theta
            return grad + d_l1 + d_l2

        # incremental learnings (as long as actual target is available)
        i = 0
        while t+fp < len(y):
            # get new data at the time 't'
            X_t, y_t = X[t] , y[t+fp]
            # predicted variable of 'fp' months ahead
            y_hats[i+fp] = np.dot(X_t, thetas[i])
            # predicted error 
            errors[i+fp] = (y_t - y_hats[i+fp])
            # define subset of X and y
            X_sub, y_sub = X[t-sc+1:t+1], y[t+fp-sc+1:t+fp+1]
            # get the theta at the time t
            theta_ = thetas[[i]].reshape(-1, 1)
            # iterations
            for _ in range(self.iter_):
                # get the gradient of MSE with elastic net regularization
                grad = gradient(X_sub, y_sub, theta_)
                # update the theta
                theta_ -= self.eta_* grad
            # finally update theta in certain future ahead
            thetas[i+fp] = theta_.reshape(1, -1)

            # increment t and idx
            t += 1
            i += 1

        # calculate the future values
        future = np.sum(np.multiply(X[-fp:], thetas[i:]), axis=1)
        
        return thetas, y_hats, errors, future


    def logistic_reg(self, X, y, t, fp, sc):
        """
        Return the following matrix (dtype: np.array)
        - "thetas"  -> parameters (intercept + coefficients) at each step
        - dictionary: predicted y values:
            - "cat" key: "y_preds_c" -> predicted labels at each step
            - "proba" ley: "y_preds_p" -> predicted probability at each step (nagetige & positive class)
        - "errors"  -> prediction errors (actual - predicted values)
        - dictionary: future y values 
            - "cat": predicted labels at each step
            - "proba": probability of the positive class label

        Parameters:
        - "X": np.array -> independent variables
        - "y": np.array -> target variables (shouold be categorical)
        - "t": int -> number of data that were used for the initial parameter creation.
        - "fp": the performance of certain months of future
        - "sc": int -> scope of the latest data for parameter updates.
        """
        # define sigmoid function
        def sigmoid(h):
            return 1 / (1 + np.exp(-h))

        # calculate the gradient vector
        def gradient(X, y, theta, w, n=sc, a_=self.alpha_, l_=self.lambda_):
            """
            X: (n, k)
            y: (n, k)
            theta: (k, 1)
            
            'n' is number of the recent observations (scopes)
            'k' is number of parameters (including bias term)
            """
            preds = sigmoid(np.dot(X, theta))
            grad = -1/n * np.dot(X.T, np.multiply(y - preds, w))
            l1 = l_ * np.sign(theta)
            l2 = (1 - l_) * theta

            return grad + (a_/n) * (l1 + l2)
        
        # initialize matrics to store values at each step
        thetas = np.full((X.shape[0]-t, X.shape[1]), np.nan)
        y_hats = np.full((X.shape[0]-t, 1), np.nan)
        errors = np.full((X.shape[0]-t, 1), np.nan)
        # initial training 
        init_X, init_y = X[0:t], y[fp:t+fp]
        logit = LogisticRegression(fit_intercept=False, class_weight='balanced')
        logit.fit(init_X, init_y.flatten())
        # set initial parameters
        thetas[:fp] = logit.coef_

        # incremental learnings (as long as actual target is available)
        i = 0
        # counting unique numbers
        y_ex_nan = y[~np.isnan(y)]
        count_labels = {k: np.sum(y_ex_nan[:t] == k) for k in np.unique(y_ex_nan)}
        while t+fp < len(y):
            # get new data at the time 't'
            X_t, y_t = X[[t]] , y[t+fp]
            # predicted variable of 'fp' months ahead
            y_hats[i+fp] = sigmoid(np.dot(X_t, thetas[i]))
            # increment counts of unique labels
            count_labels[y_t[0]] += 1
            # calculate weights
            weights = {k: sum(count_labels.values()) / (2 * v) for k, v in count_labels.items()}
            w_vect = np.vectorize(weights.get)
            # logistic loss
            y_proba = {0: 1-y_hats[i+fp], 1: y_hats[i+fp]}
            errors[i+fp] = -1 * weights[y_t[0]] * np.log(y_proba[y_t[0]])
            # define subset of X and y
            X_sub, y_sub = X[t-sc+1:t+1], y[t+fp-sc+1:t+fp+1]
            # get the current theta
            theta_ = thetas[[i]].reshape(-1, 1)
            # iterations
            for _ in range(self.iter_):
                # get the gradient
                grad = gradient(X_sub, y_sub, theta_, w_vect(y_sub))
                # update the theta
                theta_ -= self.eta_* grad
           
            # assign finialized theta
            thetas[i+fp] = theta_.reshape(1, -1)

            # increment t and idx
            t += 1
            i += 1

        # predict the future labels
        future = sigmoid(np.sum(np.multiply(X[-fp:], thetas[i:]), axis=1))

        return thetas, y_hats, errors, future 
        

    def tree_reg(self, X, y, t, fp, max_death_):
        
        # initialize matrics to store values at each step
        f_impacts = np.full((X.shape[0]-t, X.shape[1]), np.nan)
        y_hats = np.full((X.shape[0]-t, 1), np.nan)
        errors = np.full((X.shape[0]-t, 1), np.nan)
        models = []

        # initial training 
        init_X, init_y = X[0:t], y[fp:t+fp]
        tree = DecisionTreeRegressor(max_depth=max_death_)
        tree.fit(init_X, init_y.flatten())
        # set initial feature impacts
        f_impacts[:fp] = tree.feature_importances_
        # store the current data
        for _ in range(fp):
            f_impacts[:fp] = tree.feature_importances_
            models.append(deepcopy(tree))

        # incremental learning
        i = 0
        while t+fp < len(y):
            # predict y based on the model that was created certain months before
            y_hat = models[i].predict(X[[t]])
            y_hats[i+fp] = y_hat
            # calculate error
            errors[i+fp] = y[t]-y_hat
            # recreate the tree diagram
            tree.fit(X[:t+1], y[:t+1])
            f_impacts[i+fp] = tree.feature_importances_
            models.append(deepcopy(tree))

            # increment t and idx
            t += 1
            i += 1
        
        # future values
        future = np.array([models[i2].predict(X[[i2]]) for i2 in range(i, i+fp)])

        return f_impacts, y_hats, errors, future, models
    

    def evaluation(self, model, X, y, t, thetas, y_hats, errors, fp):
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
        # define feature matrix and target that were used for incremental learning
        X_, y_ = X[t:-fp], y[t+fp:]
        #  number of observations features (excluding bias)
        n, k = X_.shape[0], X.shape[1] - 1
        # assign the data frame index and columns
        rows = ['original'] + [f'theta{i}=0' for i in self.X_use_idx[1:]]

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
                    error_ = errors[fp:]
                # conduct the backward elimination
                else:
                    # copy the parameters matrix
                    theta_ = thetas[:-fp].copy()
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
                    y_hats_ = np.where(y_hats[fp:] >= 0.5, 1, 0)
                # conduct the backward elimination
                else:
                    # copy the parameters matrix
                    theta_ = thetas[:-fp].copy()
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

        elif model == 'CART':
            # names of the measure
            cols = ['rmse', 'se', 'r2', 'adj_r2']
            # sum of square total
            sst = np.sum((y_ - np.mean(y_))**2)
            errors = errors[~np.isnan(errors)]
            # Calculate Measures (rmse, se, r2, adj_r2, in order)
            sse = np.sum(errors**2)
    
            return pd.DataFrame({
                'rmse': np.sqrt((errors**2).mean()),
                'se': np.sqrt(sse / (n - k - 1)),
                'r2': 1 - sse/sst,
                'adj_r2': 1 - (sse/(n - k -1))/(sst/(n-1))
            }, index=[0])

        else:
            raise(TypeError("Model should be either one of ['LinR', 'LogR', 'CART']"))


    def compare_perf(self, model: str):
        # define measures & modify perf_df
        if model == 'CART':
            measures = list(self.perf_df[model].columns)[2:]
            perf_df = self.perf_df[model].groupby(['FP', 'MA']).mean()

        else:
            measures = list(self.perf_df[model].columns)[3:]
            perf_df = self.perf_df[model].drop('SC', axis=1).groupby(['FP', 'MA']).mean()
    
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
        n = len(self.X_use_idx) - 1
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
                m1_d['theta'] += [self.X_name[i] for i in self.X_use_idx[1:]]
                m2_d['theta'] += [self.X_name[i] for i in self.X_use_idx[1:]]
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
        if model == 'CART':
            thetas_dim = self.results[model][f'{max_fp}FP'][f'{max_ma}MA']['f_impacts'].shape
        else:

            thetas_dim = self.results[model][f'{max_fp}FP'][f'{max_ma}MA_{first_sc}SC']['theta'][:, 1:].shape
        
        min_thetas_dim = (thetas_dim[0]-max_fp, thetas_dim[1])
        # get the last date of data frame
        last_date = pd.to_datetime(self.df['Date'].iloc[-1])
        # get the next month last date
        next_month_last = last_date + relativedelta(months=1)
        # define start date
        start_date = next_month_last - relativedelta(months=min_thetas_dim[0]-1)
        # define date ranges
        date_ranges = np.array(pd.date_range(start_date, next_month_last, freq='M'))
        # define data frame
        thetas_df = pd.DataFrame({"Date": date_ranges})

        # initialize thetas data
        thetas_data = np.zeros(min_thetas_dim)
        denominator = 0
        # get all thetas from multiple models
        for k1 in self.results[model].keys():
            fp = int(k1[0])
            for k2 in self.results[model][k1].keys():
                # index start and end
                s, e = max_fp - fp+1, thetas_dim[0] - fp + 1
                if model == 'CART':
                    thetas_data = self.results[model][k1][k2]['f_impacts'][s:e]
                else:
                    thetas_data += self.results[model][k1][k2]['theta'][s:e, 1:]
                denominator += 1

        # calculate mean
        thetas_mean = thetas_data / denominator
        # define column names
        s_idx = 0 if model == 'CART' else 1
        cols = [self.X_name[i] for i in self.X_use_idx[s_idx:]]
        # adding thetas_data
        thetas_df = pd.concat([thetas_df, pd.DataFrame(thetas_mean, columns=cols)], axis=1)
        
        # create visualizations
        fig = px.line(thetas_df, 'Date', cols)
        for trace in fig.data:
            trace.hovertemplate = f'%{{y}}'

        fig.update_layout(
            title='Impact of Economic Indicators on S&P500 Over Time',
            legend=dict(title_text='Indicators', orientation='h', font_size=11, font_color='lightgray',
                        x=-0.025, y=1.05, xanchor='left', yanchor='top'),
            template='plotly_dark', hovermode="x unified",
            yaxis=dict(title_text='Strengths'),
            height=400, width=800, margin=go.layout.Margin(t=60, b=50, l=50, r=30),
            )
        
        return fig

