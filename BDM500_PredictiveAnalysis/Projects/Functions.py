import numpy as np
from numpy.typing import NDArray
import pandas as pd
import re

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import PolynomialFeatures


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
    

class Regression:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # initialize other variables
        self.X_name = []
        self.y_name = ''
        self.ma_opts = []
        self.fp_opts = []
        self.datasets = {}

        self.scopes = []
        self.results = {}
        self.be_tests = {'ma': {}, 'sc': {}}
        self.perf_df = pd.DataFrame(data=[], columns=['SC', 'MA', 'FP', 'RMSE', 'SE', 'R2', 'Adj-R2'])

        # initialize class
        self.poly = None
        self.scaler = None
        
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
            X_ss = self.scaler.fit_transform(120, X_poly[ma_i:, 1:])
            # add bias term
            X_ss_b = np.c_[np.ones(X_ss.shape[0]), X_ss]

            # apply data shift
            for fp_i in self.fp_opts:
                self.datasets.update({f"{ma_i}MA_{fp_i}M": {'X': X_ss_b, 'y': y_ma_i[fp_i:]}})


    def model_result(self, scopes: list, model_name: str = '', eta: float = 0.01, alpha: float = 1.0, lambda_: float = 0.5):
        # set spaces
        self.be_tests['ma'].update({ma: [] for ma in self.ma_opts})
        
        for i, s in enumerate(scopes):
            # set spaces
            self.results[s] = {}
            self.be_tests['sc'].update({s: []})

            # each dataset
            for j, d_key in enumerate(self.datasets.keys()):
                idx = i * len(self.datasets.keys()) + j
                data = self.datasets[d_key]
                theta, y_hat, error = self.gradient_descent(
                    X=data['X'], y=data['y'], t=120, s=s, 
                    eta=eta, alpha=alpha, lambda_=lambda_
                )
                # store all data
                self.results[s].update({d_key: {'theta': theta, 'y_hat': y_hat, 'error': error}})
                # get test result of backward elimination
                be_test_df = self.evaluation(data['X'], data['y'], 120, theta, y_hat, error)
                # retrienve only performance without any changes in each coefficient
                ma, fp = d_key.split('_')
                ma_int = int(re.findall(r'\d+', ma)[0])
                self.perf_df.loc[idx] = [s, ma_int, fp] + list(be_test_df.iloc[0])
                # store its result as NDArray
                self.be_tests['sc'][s].append(np.array(be_test_df))
                ma_idx = self.ma_opts[j // len(self.fp_opts)]
                self.be_tests['ma'][ma_idx].append(np.array(be_test_df))

        
        self.compere_perf_fig = self.compare_perf(model_name)
        self.be_test_sc_fig = self.backward_elimination('sc')
        self.be_test_ma_fig = self.backward_elimination('ma')

        return self.compere_perf_fig, self.be_test_sc_fig, self.be_test_ma_fig




    def detail_perf(self, ma: int, fp: int, sc: int):
        """
        Return two plotly chart:
        - Line chart: Comparing the predicted and actual value.
        - Mix chart:  Distribution of the predicted errors by histogram and scatter plots.

        Return: plotly.graph_objs._figure.Figure

        """
        # get model and data
        d_name = f'{ma}MA_{fp}M'
        data = self.datasets[d_name]
        theta, y_hat, error = tuple(self.results[sc][d_name].values())
        
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

        # future value
        if fp > 0:
            y, m = x_date[-1].year, x_date[-1].month - 1
            months = [(i % 12) + 1 for i in range(m, m+fp+1)]
            add_on = []
            for i, month in enumerate(months):
                # update the year
                if i != 1 and month == 1:
                    y += 1
                add_on.append(str(pd.Timestamp(y, month, 1) + pd.offsets.MonthEnd(1)).split()[0])
            new_X = data['X'][num_obs:]
            future = np.dot(new_X, theta[-1].reshape(-1,1)).flatten()
            fig1.add_trace(go.Scatter(
                x=pd.to_datetime(add_on), y=np.concatenate((y_hat[-1], future)), 
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
                    marker=dict(color='red', opacity=0.5), name='Error'),
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


    def gradient_descent(self, X, y, t, s, eta=0.01, alpha=1.0, lambda_=0.5):
        """
        Return the following three matrix (dtype: np.array)
        - "theta"  -> parameters (intercept + coefficients) at each step
        - "y_hats" -> predicted values at each step
        - "error"  -> prediction errors (actual - predicted values); SSE

        Parameters:
        - "X": np.array -> independent variables
        - "y": np.array -> target variables
        - "t": int -> number of data that were used for the initial parameter creation.
        - "s": int -> scope of the latest data for parameter updates 
        - "eta": learning rate for gradient descent
        - "alpha": how strength the regularization is.
        - "lambda_": balancing between ridge and lasso regularization.

        Brief Steps:
        - Initialize matries for theta, y_hats, error.
        - Apply a given number of data ("t") to the mutiple linear regression (normal equation).
        - Define the initial parameters from the trained model.
        - At each step (total steps are len(y) - t):
            - Get a single pair of unfamilar data; both X and y.
            - Predict the target ("y_hats") based on the latest parameters("theta[i]").
            - Calculate the difference between actual and predicted values; "error[i]".
            - Update parameters for the next step ("theta[i+1]").
        """

        # define all matrix to be returned
        k = len(self.X_name) # num of features + bias
        theta = np.zeros((len(y)-t + 1, k))
        y_hats = np.zeros((len(y)-t, 1))
        error = np.zeros((len(y)-t, 1))
        # Modify the matrix of features; adding bias

        # define elastic net derivative
        def elastic_net_der(theta, X, y, n=s, a_=alpha, l_=lambda_):
            # reshape
            X = X.reshape(n, -1)
            y_hat = np.dot(X, theta).reshape(-1, 1)
            # error weighted feature
            ewf = 2/n * np.dot(X.T, y_hat - y).reshape(1, -1)
            d_l1 = l_ * a_ * np.sign(theta)
            d_l2 = (1 - l_) * a_ * theta
            return ewf + d_l1 + d_l2
        
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
            error[i] =  (y_hats[i] - y_i)
            # start gradient descent based on the data scope ("s")
            # if scope is > 1, taking care of the predicted error from the recent data over a given scope ('s')
            if s > 1:
                # get the latest data based on the scope ('S')
                X_, y_ = X[idx-s+1:idx+1], y[idx-s+1:idx+1]
                derivative = elastic_net_der(theta[i], X_, y_)
                theta[i+1] = theta[i] - eta * derivative

            # if scope is 1, only taking care of the predicted error from the most recent data 
            else:
                derivative = elastic_net_der(theta[i], X_i, y_i)
                theta[i+1] = theta[i] - eta * derivative
                
        return theta, y_hats, error


    def evaluation(self, X, y, t, theta, y_hats, error):
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
            - "error": difference between actual and predicted values at each step

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
            y_mean = np.mean(y[t:])
            #  sum of square total
            sst = np.sum((y[t:] - y_mean)**2)
            #  define feature matrix and target based on "t"
            X_, y_ = X[t:n+t], y[t:]

            #  number of coefficients
            for i in range(k+1):
                # simply applying the given error
                if i == 0:
                    error_ = error
                # conduct the backward elimination
                else:
                    # copy the parameters matrix
                    theta_ = theta[:n].copy()
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


    def compare_perf(self, model_name: str = ''):
        """
        Comparing the performance of linear regression models.

        Return: plotly.graph_objs._figure.Figure

        Parameters:
        - "perf_df": keys show each model; values show pd.DataFrame.
        - "model_name": model name to show it on the figure title
        """
        # define customer function
        def upper_error(x):
            return x.max() - x.mean()

        def lower_error(x):
            return x.mean() - x.min()

        # deine measures
        measures = list(self.perf_df.columns)[3:]
        # modity perf_df
        perf_df = self.perf_df.drop('FP', axis=1).groupby(['SC', 'MA']).agg(['mean', upper_error, lower_error])
        perf_df.columns = ['_'.join(col) for col in perf_df.columns]
        perf_df = perf_df.reset_index()
        perf_df['SC'] = perf_df['SC'].astype(str)
        
        # define fig
        fig = make_subplots(rows=2, cols=2, subplot_titles=measures,
                            horizontal_spacing=0.05, vertical_spacing=0.05,
                            shared_xaxes=True, shared_yaxes=True)
        [fig.layout.annotations[i].update(font=dict(size=11, color='grey')) for i in range(len(measures))]

        for i, ms in enumerate(measures):
            # row and col idx
            r, c = i // 2, i % 2
            # figure data
            fig_data = px.scatter(
                perf_df, x='MA', y=f'{ms}_mean', color='SC',
                error_y=f'{ms}_upper_error', error_y_minus=f'{ms}_lower_error'
            )
            # add each data
            for f_data in fig_data.data[:4]:
                # remove duplicated legends
                f_data.showlegend = True if i == 0 else False
                fig.add_trace(f_data, row=r+1, col=c+1)
            
        main_title = f'Comparing the {model_name} Model Results on Different Conditions'
        sub_title1 = f'<br><span {self.SUB_CSS}> -- Scopes: how many month of the latest data is used for parameter adjustments.</span>'
        sub_title2 = f'<br><span {self.SUB_CSS}> -- Error Bars: Showing mean, min and max of each measures among various future predictions.</span>'

        # edit layput
        fig.update_xaxes(title_text='Moving Averages', row=2)
        fig.update_layout(height=500, width=800, template='plotly_dark', 
                        title=dict(text=main_title + sub_title1 + sub_title2,  yanchor="top", y=0.95),
                        legend=dict(title_text='Scopes', title_font={'color':'lightgrey'}),
                        margin=go.layout.Margin(t=100, l=40, r=40))

        return fig


    def backward_elimination(self, type_: str):
        """
        Evaluate the backward elimination for all models with focus on RMSE and adjusted R2.
        Visualize the scatter plots to show the difference from original result

        Parameter:
        - 'type': either one of 'sc' or 'ma'
        """
        be_test = self.be_tests[type_]
        # number of features; excluding bias term
        n = len(self.X_name) - 1
        rmse_dict = {type_: [], 'theta': [], 'diff': []}
        r2_dict = {type_: [], 'theta': [], 'diff': []}

        for key in be_test:
            for matrix in be_test[key]:
                # learning method
                rmse_dict[type_] += [key] * (n)
                r2_dict[type_] += [key] * (n)
                # add theta name
                rmse_dict['theta'] += list(self.X_name[1:])
                r2_dict['theta'] += list(self.X_name[1:])
                # first column is RMSE and last one is adjusted R2
                diff = matrix[1:] - matrix[0]
                # add error
                rmse_dict['diff'] += list(diff[:, 0])
                r2_dict['diff'] += list(diff[:, -1])
                

        # plot data points
        rmse_df = pd.DataFrame(rmse_dict)
        r2_df = pd.DataFrame(r2_dict)
        fig1 = px.strip(data_frame=rmse_df, x='theta', y='diff', color=type_)
        fig2 = px.strip(data_frame=r2_df, x='theta', y='diff', color=type_)

        # set figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=["RMSE", "Adj-R2"])
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
        be_type = 'Scopes' if type_ == 'sc' else 'Moving Averages'
        main = f"Observe Backward Elimination in All Models With Repect to {be_type}" 
        sub = f"<br><span {self.SUB_CSS}> -- How meansures are changed by removing the impact of each coefficient</span>"
        fig.update_layout(height=400, width=800, template='plotly_dark', 
                        title_text=main + sub, yaxis_title="Difference",
                        legend=dict(title_text=f'{be_type}:', orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                        margin=go.layout.Margin(t=80, b=60, l=80, r=40))
    
        return fig