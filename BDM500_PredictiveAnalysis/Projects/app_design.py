from dash import Dash, html, dash_table, dcc, Output, Input, State, ALL, MATCH, ctx, Patch, no_update
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go 

import numpy as np
import pandas as pd
from scipy.stats import skew
from itertools import combinations

import os
import json

from Functions import PredictiveAnalysis

# dash table design
TABLE_STYLE = dict(
    style_cell={
        'fontFamily': 'Segoe UI',
        'textAlign': 'left',
        'padding': '10px',
        'border': '1px solid #444'
    },
    style_header={
        'fontWeight': 'bold',
        'backgroundColor': 'rgb(30, 30, 30)', 
        'color': 'white',
        'border': '1px solid #444',
    },
    style_data={
        'backgroundColor': 'rgb(50, 50, 50)', 
        'color': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(70, 70, 70)',
        },
        {
            'if': {'column_id': 'link'},
            'color': '#59afe1',
            'cursor': 'pointer',
            'textDecoration': 'underline',
        },

        {
            'if': {'state': 'active'}, 
            'backgroundColor': 'rgb(65, 65, 65)',
            'border': '1px solid #888',
        },
    ],
    style_table = {
        'overflowX': 'auto'
    }
)
# common button style
BUTTON_STYLE = dict(
    backgroundColor='#217CA3',
    color='white',
    border='none',
    padding='10px 20px',
    margin='10px',
    borderRadius='5px',
    cursor='pointer',
    fontSize='16px',
    width='100%',
)
#  plotly figure design
FIG_LAYOUT = dict(
    template='plotly_dark',
    yaxis=dict(ticksuffix=" "*2),
    yaxis2=dict(tickprefix=" "*2),
)
# horizontal dash line
DASH_LINE = html.Hr(style={'borderTop': '2px dashed #fff', 'margin': '75px 0'})
S_DASH_LINE = html.Hr(style={'borderTop': '1px dashed #fff', 'margin': '25px 10px'})
# sub title CSS
SUB_CSS = 'style="font-size: 12.5px; color: lightgrey;"'

# color
COLORS = [
    'rgba(99, 110, 250, 0.70)',
    'rgba(239, 85, 59, 0.70)',
    'rgba(0, 204, 150, 0.70)',
    'rgba(171, 99, 250, 0.70)',
    'rgba(255, 161, 90, 0.70)',
    'rgba(25, 211, 243, 0.70)',
    'rgba(255, 102, 146, 0.70)',
    'rgba(182, 232, 128, 0.70)',
    'rgba(255, 151, 255, 0.70)',
    'rgba(254, 203, 82, 0.70)'
]
# extract color from fig.data
def FIG_COLOR(fig_data):
    #if len(fig_data) > 1:
    #        fig_data = fig_data[0]

    for name in ['marker', 'line', 'bar']: 
        attribute = getattr(fig_data, name, None) 
        if attribute:
            color = getattr(attribute, 'color', None)
            if color:
                return color
            
    return False


class Design:
    def __init__(self, app: Dash, df: pd.DataFrame, path: str):
        # app
        self.app = app
        # assign original data frame
        self.origin = df
        # initialize id
        self.design_ha_figs_id = 0

        # Data Process
        #  for preprocessing
        self.df1 = self.origin.dropna()
        self.group_year = self.df1.groupby('Year').count().reset_index()
        self.df2 = self.df1.copy()
        self.df2 = self.df2[self.df2['Year'] >= 1978].reset_index(drop=True)

        #  for observation
        self.df3 = self.df2.copy()
        #  copy the current SP500 as SP500_Price
        self.df3['SP500_Price'] = self.df3['SP500'].loc[:]
        #  whether the S&P500 rises (1) or falls(0) from the previous year
        cat_values = (self.df3['SP500_Price'] > self.df3['SP500_Price'].shift(12)).astype(int)
        cat_values[:12] = None
        self.df3['SP500_Rise'] = cat_values
        #  changes to %YoY
        chg_YoY = ['CPI', 'CSENT', 'IPM', 'HOUSE', 'SP500']
        self.df3.loc[:, chg_YoY] = self.df3[chg_YoY].pct_change(12) * 100
        #  drop the rows with missing values
        self.df3.dropna(inplace=True)
        self.df3.reset_index(drop=True, inplace=True)
        #  correlation matrics (1)
        self.corr1 = self.df3[['SP500', 'MY10Y', 'CPI', 'CSENT', 'IPM', 'HOUSE', 'UNEMP']].corr().reset_index(names='')
        # add new data
        self.df3.insert(loc=9, column='LRIR', value=self.df3['MY10Y'] - self.df3['CPI'])
        # drop MY10Y and CPI
        self.df3.drop(['MY10Y', 'CPI'], axis=1, inplace=True)
        # show new correlation matrix
        self.features = ['SP500', 'CSENT', 'IPM', 'HOUSE', 'UNEMP', 'LRIR']
        self.corr2 = self.df3[self.features].corr().reset_index(names='')
        
        # change to log(UNEMP)
        self.df4 = self.df3.copy()
        self.df4['UNEMP'] = np.log(self.df3['UNEMP'])

        # assign PredictiveAnalysis class
        self.PA = PredictiveAnalysis(self.df4)
        self.new_df = self.PA.create_data(['CSENT', 'IPM', 'HOUSE', 'UNEMP', 'LRIR'], 'SP500', ma=[1,2,3], fp=[1,2,3,4,5,6], init_train=120, poly_d=1)

        # loading json files
        self.fig_data = {}
        f_names = ['default_linear', 'default_logit', 'default_cart', 'final_linear']
        for i, k1 in enumerate(['LinR', 'LogR', 'CART', 'LinBest']):
            # set space
            self.fig_data[k1] = {}
            # loading json files
            file_name = os.path.join(path, 'json_files', f"{f_names[i]}.json")
            with open(file_name, 'r') as file:
                data_dict = json.load(file)
                for k2, v2 in data_dict.items():
                    # Deserialize DataFrame
                    if k2 == 'pred_df':  # 'pred_df' is the key for the DataFrame
                        self.fig_data[k1][k2] = pd.read_json(v2)

                    # Deserialize Plotly Figures
                    else:
                        self.fig_data[k1][k2] = go.Figure(json.loads(v2))

    # Contents
    def header(self):
        """
        Create the header of the app
        """
        header = dbc.Navbar(
            [
                dbc.Container(
                    [
                        # Title
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(dbc.NavbarBrand("S&P500 Predictive Analysis", className="ml-2")),
                                ],
                                align="center",
                            ),
                            href="/",
                        ),
                    ],
                    fluid=True,
                ),
            ],
            color="dark",
            dark=True,
            style={"height": "100px", 'marginBottom': '20px'},
        )

        return header

    def overview(self):
        """
        Show Project overview, Research questions (objectives), and Expected Concerns
        """
        # main title
        main_t = ['Project Overview', 'Objectives / Questions', 'Expected Concerns']
        # sub titles
        t1 = ['Main Theme', 'Intended Audiences', 'Applications']
        t2 = ['Various Aspects', 'Facter Analysis', 'Model Performance']
        t3 = ['Distribution of Data', 'Lag of Data Release', 'Market Volatility', 'Complex Relationship']
        # comments
        c1 = [
                'Building incrementally-learning predictive models for the S&P500 index using macroeconomic indicators.',
                'Stakeholders in finance, economics, and stock market, particularly in mutual funds, investment banks, and pension funds​.',
                'Guiding investment decisions, Informing risk management strategies, Analysis for traders and institutional investors',
        ]
        c2 = [
                'How will the future performance of the S&P 500 index be changed by different conditions; moving averages, scopes, future predictions, and different models?',
                'Which economic indicators are likely to significantly affect the future performance of the S&P500 index, and how those impacts have been changed over time?',
                'How accurately can each model predict the performance of the S&P500 index from one to six months ahead based on the latest economic indicator?'
        ]
        c3 = [
                """
                    Economic indicators often do not follow a normal distribution.
                    For example, the unemployment rate tends to be right-skewed, even after transformations
                """,
                """
                    Economic data is reported with a time lag of about 1 month.
                    Real-time data, like S&P500, differs from economic data, causing timing challenges.
                """,
                """
                    Human emotions impact market decisions, leading to unpredictable market behaviors.
                    Market volatility and randomness pose challenges in explaining market actions.
                """,
                """
                    The market is influenced by complex economic dependencies, disruptions, and geopolitical tensions.
                    These factors hinder the development of accurate predictive models.
                """,
        ]
        # types of comments
        types_ = ['P', 'P', 'Ul']

        # all cards
        cards = [
            html.Div([
                html.H3(main, style={'textAlign': 'center', 'margin': '30px auto'}),
                self.design_cards_equal(sub, c, type_),
                DASH_LINE,
            ])
        for main, sub, c, type_ in zip(main_t, [t1, t2, t3], [c1, c2, c3], types_)
        ]
        
        return dbc.Container(cards)

    def dataset(self):
        """
        Structure of the dataset name, dtype, and description.
        """
        # all dataset info
        dataset = [
            {"Symbol": "Date", "Dtype": "string", "Description": "End dates of each month; format is '%Y-%m-%d'", "Data Source":""},
            {"Symbol": "Year", "Dtype": "int", "Description": "Year of date by integer type", "Data Source": ""},
            {"Symbol": "Month", "Dtype": "int", "Description": "Month of date by integer type", "Data Source": ""},
            {"Symbol": "SP500  *1*2", "Dtype": "float", "Description": "S&P 500 Index (ticker code: ^GSPC)", "Data Source": "[Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/)"},
            {"Symbol": "MY10Y  *2", "Dtype": "float", "Description": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity", "Data Source": "[Fred](https://fred.stlouisfed.org/series/DGS10)"},
            {"Symbol": "CPI  *1*3", "Dtype": "float", "Description": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average", "Data Source": "[Fred](https://fred.stlouisfed.org/series/CPIAUCSL)"},
            {"Symbol": "CSENT  *1*3", "Dtype": "float", "Description": "University of Michigan: Consumer Sentiment", "Data Source": "[Fred](https://fred.stlouisfed.org/series/UMCSENT)"},
            {"Symbol": "IPM  *1*3", "Dtype": "float", "Description": "Industrial Production in Manufacturing", "Data Source": "[Fred](https://fred.stlouisfed.org/series/IPMAN)"},
            {"Symbol": "HOUSE  *1*3", "Dtype": "float", "Description": "New One Family Houses Sold", "Data Source": "[Fred](https://fred.stlouisfed.org/series/HSN1F)"},
            {"Symbol": "UNEMP  *3", "Dtype": "float", "Description": "Unemployment rate", "Data Source": "[Fred](https://fred.stlouisfed.org/series/UNRATE)"},
            {"Symbol": "LRIR", "Dtype": "float", "Description": "Long-term Real Interest Rate; the subtraction of MY10Y by %YoY CPI", "Data Source": ""}
        ]
        # notifications
        notifications = [
            "*1: The data will be converted to the Year-over-Year (YoY) percent growth rate.",
            "*2: Data in each row shows the closing value on the last trading day of each month.",
            "*3: Data in each row is adjusted by the actual result of each month rather than the released month."
        ]

        df = pd.DataFrame(dataset)

        table = [
            html.H3("Dataset Information", style={'textAlign': 'center', 'margin': '30px auto'}),
            # Table
            dash_table.DataTable(
                id='table',
                columns=[
                    {"name": i, "id": i} if i != 'Data Source' else {"name": i, "id": i, "type": "text", "presentation": "markdown"} for i in df.columns
                ],
                data=df.to_dict('records'),
                **TABLE_STYLE
            ),
            # Notifications section
            html.Div(
                [html.P(note, style={'color': '#eeeeee', 'fontSize': '0.85em'}) for note in notifications],
            style={'backgroundColor': '#333333', 'padding': '15px', 'margin': '20px 0', 'borderRadius': '5px'}
            ),
            DASH_LINE,
        ]

        return dbc.Container(table)

    def data_preprocessing(self):
        """
        Showing the data preprocessing phases step by step.
        (1): Addressing Incompleteness in Time Series
        (2): Data Modification / Feature Selection
        """
        elements = [
            html.H3('Data Preprocessing', style={'textAlign': 'center', 'margin': '30px auto'})
        ]

        # (1) Dealing with Incomplete Data
        #  graph
        fig = px.line(self.group_year, x='Year', y='Month', title='Number of monthly data on each year')
        fig.update_layout(
            FIG_LAYOUT,
            hovermode="x unified",    
        )
        # graph observations 
        comments_1 = """
            Figure shows the number of data on each year after removing all missing data.
            The first several years of data are still incompleted; missing data in some months.
            In this project, the consistency of the time series data is critical because moving averages are applied to target variables.
            In the incremental learning phases, the previous month of data will affect the next parameter setup.
            Hence, those incompleted data will be removed to maintain the time series consistency.
        """ 
        elements += [
            dbc.Row([
                html.H4("(1): Addressing Incompleteness in Time Series", style={'margin': '0 0 30px'}),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id='example-graph',
                            figure=fig,
                            style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
                        ), 
                        width=5
                    ),
                    dbc.Col([
                        # html.Div for graph observations
                        self.design_observe(comments_1, 'Ul'),
                        dbc.Row([
                            html.H5("Data Preview (After Modification)", style={'color': '#d9d9d9', 'margin': '15px'}),
                            dash_table.DataTable(
                                data=self.df2.head().round(2).to_dict('records'),
                                columns=[{'name': i, 'id': i} for i in self.df2.columns],
                                **TABLE_STYLE,
                            ),
                        ], style={'marin': '15px'})
                    ], width=7
                    ),
                ], justify='center', align='center'),
                S_DASH_LINE
            ])
        ]

        # (2) Data Modification
        titles = [
            'Categorical Value',
            'Year-over-Year Growth',
        ]
        contents = [
            'Generate a categorical value; whether the S&P500 index rises("1") or falls("0") compared to the same month of the previous year.',
            'Converting to the Year-over-Year (YoY) Percent Growth, which is subject to CPI, CSENT, IMP, HOUSE, and SP500.',
        ]

        elements += [
            dbc.Row([
                html.H4("(2): Modifying the Data / Creating Categorical Labels", style={'margin': '0 0 30px'}),
                dbc.Row([
                    self.design_cards_equal(titles, contents),
                    html.H5("Data Preview (After Modification)", style={'color': '#d9d9d9', 'margin': '15px'}),
                    dash_table.DataTable(
                        data=self.df3.head().round(2).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in self.df3.columns],
                        **TABLE_STYLE,
                    ),
                ]),
                DASH_LINE
            ])
        ]

        return dbc.Container(elements)

    def data_observation(self):
        # store all elements
        elements = [html.H3('Data Observations', style={'textAlign': 'center', 'margin': '30px auto'})]

        ##### (1) Correlation Matrix & Feature Selection
        # setting tabs for two matrics
        tab_titles = ['Original Corr', 'New Corr']
        tab_elements = [
            dash_table.DataTable(
                data=self.corr1.round(2).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in self.corr1.columns],
                **TABLE_STYLE,
            ),
            dash_table.DataTable(
                data=self.corr2.round(2).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in self.corr2.columns],
                **TABLE_STYLE,
            ),
            ]
        # observations
        comments1_1 = """
            CSENT, IPM, and HOUSE are positively correlated with the S&P500 index, and those relationships are relative high.
            MY10Y, CPI, and UNEMP have weaker relationship with the S&P500 index; all values are close to zero.
            MY10Y and CPI shows the strongest correlation; although there is less risk of multicollinearity, a derived data "LRIR" will be created for feature removal.
        """ 
        comments1_2 = """
            Symbol: LRIR
            Name: Longer-term Real Interest Rate
            Calculation: MY10Y - CPI(YoY growth)
            Description: Assuming that the nominal rate can be reflected from MY10Y, and the inflation rate is the YoY percent change in CPI.
        """
        # output elements
        elements += [
            dbc.Row([
                html.H4("Correlation Matrix & Feature Selection", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_tabs(tab_titles, tab_elements, margin='0 0 20px'),
                dbc.Row([
                    dbc.Col(self.design_observe(comments1_1, type_='Ul', title='Onservations'), width=6),
                    dbc.Col(self.design_observe(comments1_2, type_='Ul', title='New Derived Data'), width=6),
                ])
            ]),
            DASH_LINE
        ]

        
        ##### (2) Skewness
        # observations
        comments_2 ="""
            IPM shows a strong left skewness, although the overall shape of the distribution is close to the normal distribution.
            On the other hand, UMEMP shows a strong right skewness; the distribution shape is substantially different from a normal distribution.
            To handle imbalanced data, the log scale is applied to UNEMP data; The right graph shows the histogram of log scaled UNEMP data; the distribution shape is not close to the normal distribution, and but the skewness was mitigated.
            Since two economic data, UNEMP and LRIR, are not applied by the YoY growth changes (the original math unit is already a percentage), the shape of those distributions might not be close to normal distribution. This is one of the expected concerns and challenges that was mentioned in the introduction part.
        """
        # fig style
        fig2_styles = dict(
            FIG_LAYOUT,
            hovermode="x unified",
            title=dict(x=0.5),
            xaxis_title="", yaxis_title="",
            height=300, width=400,
            margin=dict(t=60, l=40, r=30, b=30)
        )

        # figures
        figs2 = []
        for d in self.features:
            skewness = skew(self.df3[d])
            fig = px.histogram(self.df3, x=d, title=d+f' (skew: {skewness:.2f})', hover_data={d: False},
                               color_discrete_sequence=[COLORS[0]])
            fig.update_layout(fig2_styles)
            figs2.append(fig)

        skew_log_UNEMP = skew(self.df4['UNEMP'])
        hist2 = px.histogram(self.df4, x='UNEMP', hover_data={'UNEMP': False},
                             title=f'Log Scale of UNEMP (skew:{skew_log_UNEMP:.2f})',
                             color_discrete_sequence=[COLORS[0]])
        hist2.update_layout(fig2_styles)
        
        # add element
        elements += [
            dbc.Row([
                html.H4("Histogram / Skewness", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_ha_figs(figs=figs2)
            ]),
            dbc.Row([
                dbc.Col([
                    self.design_observe(comments_2, type_='Ul', title='Observations for Distribution')
                    ], width=8),
                dbc.Col([
                    dcc.Graph(figure=hist2)
                ], width=4)
            ], style={'alignItems': 'center', 'justifyContent': 'center'}),
            DASH_LINE
        ]


        ##### (3) Scatter plots
        # observations
        comments_3_1 = """
            There is a positive linear relationship (despite wider bands) between the SP500 index and three economic indicators (CSENT, IPM, and HOUSE).
            CSENT are positively correlated with other features, which means that the consumer sentiment data could be important factor of other economic data.
            A possible negative correlation could be found between IPM and UNEMP; the more people are working, the higher productions are.
        """
        comments_3_2 = """
            Several indicators show interesting insights into SP500. Also, those observations could be helpful to implement the robust tree algorithm.
            When CSENT or IPM declined by over 10% or 20% from a year before, respectively, SP500 is likely to be below the level of the previous year.
            When HOUSE is above 25% regardless of what kinds of economic indicators as the other axis, SP500 is likely to rise from the previous year.
            When the IPM and HOUSE decline simultaneously, SP500 will be affected negative impact. 
            Hence, three economic indicators, CSENT, IMP, and HOUSE, may have a significant impact on whether the SP500 rises or falls on YoY growth base.
        """
        # figure style
        fig3_styles = dict(
            FIG_LAYOUT,
            showlegend=False,
            title=dict(x=0.5),
            xaxis_title="", yaxis_title="",
            height=300, width=400,
            margin=dict(t=50, l=40, r=30, b=30)
        )

        # figures
        figs3_1 = []
        figs3_2 = []
        pairs = list(combinations(self.features, 2))  # get all pairs of combinations
        for d in pairs:
            fig1 = px.scatter(self.df4, x=d[0], y=d[1], title=f'{d[0]}(x) vs {d[1]}(y)', 
                             color_discrete_sequence=[COLORS[0]])
            fig1.update_layout(fig3_styles)
            figs3_1.append(fig1)

            if d[0] != 'SP500':
                fig2 = px.scatter(
                    self.df4, x=d[0], y=d[1], title=f"{d[0]}(x) vs {d[1]}(y)",
                    color=self.df4['SP500_Rise'].astype('category'), labels={'color': 'SP500'},
                    color_discrete_map={1.0: 'rgba(0, 204, 150, 0.60)', 0.0: 'rgba(239, 85, 59, 0.60)'})
                fig2.update_layout(fig3_styles)
                figs3_2.append(fig2)

        # define tabs
        tab_elements_2 = [
            [
                self.design_ha_figs(figs=figs3_1),
                self.design_observe(comments_3_1, type_='Ul', title='Feature Relationships')
            ],
            [
                self.design_ha_figs(figs=figs3_2, legends=True),
                self.design_observe(comments_3_2, type_='Ul', title='Feature Relationships')
            ]
        ]

        # add elements
        elements += [
            dbc.Row([
                html.H4("Scatter Plots Among Features", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_tabs(['Normal', 'Stratified'], tab_elements_2, margin='0 0 20px'),
            ]),
            DASH_LINE
        ]

        
        ##### (4) Trends of Economic data
        # observations
        comments_4 = [
            'There are a lot of periods when the CSENT and SP500 moved together. In 2010 and 2022, the consumer sentiment bottomed in advance before SP500.',
            'IPM has been correlated to SP500 historically. Once IPM rebounded, SP500 is highly likely to record a strong reversal in the short periods.',
            'HOUSE also correlated to SP500. If housing purchases cause the renewal of furniture and home appliances, the rise in consumption might lead to SP500 higher.',
            'The bottom of SP500 is likely to occur right before the top of UNEMP historically; both data could be inversely related to each other.',
            'In several SP500 bottoms (2001-2002, 2008-2009, and 2022-2023), LRIR was likely to move forword, which imply to be caused the fall in CPI or rise in MY10Y.'
        ]
        # figure style
        fig4_styles = dict(
            FIG_LAYOUT,
            title=dict(text='', x=0.5, y=0.9),
            height=350, width=700, hovermode='x unified',
            yaxis2=dict(overlaying='y', side='right', showgrid=False, zeroline=False),
            legend=dict(x=0.025, y=0.025, traceorder='normal', orientation='h',
                        xanchor='left', yanchor='bottom'),
            margin=dict(t=60, l=40, r=40, b=40)
        )

        # figures
        figs4 = []
        for d in self.features[1:]:  # only economic indicator
            fig = go.Figure()
            fig4_styles['title']['text'] = f'{d} vs SP500'
            fig.update_layout(fig4_styles)
            fig.add_traces([
                go.Scatter(x=self.df4['Date'], y=self.df4[d], mode='lines', name=d, line=(dict(color=COLORS[0]))),
                go.Scatter(x=self.df4['Date'], y=self.df4['SP500'], mode='lines', name='SP500', line=dict(color='rgba(200, 200, 200, 0.5)'), yaxis="y2")
            ])
            figs4.append(fig)
            
        # add elements
        elements += [
            dbc.Row([
                html.H4("Trends of Economic data", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_ha_figs(figs=figs4, comments=comments_4),
            ]),
            DASH_LINE
        ]


        ##### (5): Strafitied Histogram
        # comments
        comments_5 = """
            CSENT is higher than 10%, SP500 is likely to increase compared to the previous year; otherwise, the probability of falling rises.
            When IPM plunged from the previous year (e.g., 5% or more decline), SP500 is also likely to fall compared to the previous year.
            YoY growth of HOUSE is less than 20%, SP500 is likely to decline compared to the previous year; otherwise, SP500 rose in almost all cases.
        """
        # fig style
        fig5_styles = dict(
            FIG_LAYOUT,
            showlegend=False,
            hovermode="x unified",
            title=dict(x=0.5),
            xaxis_title="", yaxis_title="",
            height=300, width=400,
            margin=dict(t=50, l=40, r=30, b=30)
        )

        # figure
        figs5 = []
        for d in self.features[1:]:
            fig = px.histogram(
                self.df4, x=d, color='SP500_Rise', title=d,
                hover_data={d: False, 'SP500_Rise': False},
                color_discrete_map={1.0: 'rgba(0, 204, 150, 0.70)', 0.0: 'rgba(239, 85, 59, 0.70)'})
            fig.update_layout(fig5_styles)
            figs5.append(fig)

        # add element
        elements += [
            html.H4("Stratified Histogram", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            self.design_ha_figs(figs=figs5, legends=True),
            self.design_observe(comments_5, type_='Ul'),
            DASH_LINE
        ]

        return dbc.Container(elements)

    def model_descript(self):
        # core theories
        #  titles
        ct_titles = [
            'Incremental Learning', 
            'Scopes (SC)', 
            'Future Prediction (FP)', 
            'Moving Averages (MA)', 
            'Rolling Standardization'
        ]
  
        #  descriptions
        ct_descripts = [
            """
                The first 10 years of data are trained for defining the initial set of parameters.
                Model parameters are updated by the gradient descent approach of learning the rest of the data one by one.
                This incremental learning ensures enough opportunities to evaluate the predictive performance at each time.
            """,
            """
                Core idea of "Scope (SC)" is how much recent data will be used to adaptively update parameters.
                If SC is 9 (months), new parameters are adjusted so that the predicted errors from nine months of recent data are minimized.
                The longer SC is set, the more generalized the model could be, but the more likely the model could miss the latest market moves.
            """,
            """
                Future Prediction (FP) implies how much months the model will predict SP500 growth based on the latest data.
                If FP is 6 (months), the model attempts to predict the growth of SP500 after six months.
                The longer FP is set, the lower the predictive performance the model could have.
            """,
            """
                Three types of moving averages (from one to three months) are applied to the target variable "SP500".
                One moving average indicates normal value; nothing smoothing technique is applied.
                From those three numerical variables, create three categorical data; "1" for positive and "0" for negative variables. 
            """,
            """
                All economic indicators (independent variables) are appiled rolling standardization.
                First 10-year of data are normally standardized by the mean and standard deviation(std).
                The rest of data are standardized by the mean and std of the most recent 10-year of historical data.
            """
        ]

        # models
        #  titles
        m1_titles = [
            'Linear Regression / Logistic Regression', 
            'Classification and Regression Tree (CART)'
            ] 
        #  descriptions
        m1_descripts = [
            """
                Applying all five economic indicators
                Employing incremental learning approach (Online Learning)
                Testing backward elimination
                Evaluating by four regression / five classification metrics
            """,
            """
                Applying all five economic indicators
                Learning all observations at each incremental step (Offline Learning)
                Evaluating by four regression metrics
            """
            ]
        
        # model metrics
        m2_titles = ['Regression Metics', 'Classification Metrics']
        m2_descript = [
            """
                Root Mean Square (RMSE): How much errors could occur between the predicted prices and the actual ones.
                Standard Error of Estimate (SE): How much variation could occur in the actual target based on the same condition of independent variables.
                Coffeficient of Determination (R2): How well the regression model explains the variation of a target value.
                Adjusted R2 (Adj-R2): R2 with the penalty for the number of independent variables.
            """,

            """
                Accuracy: How the model can correctly predict the target values.
                Precision: How the model can avoid false positives.
                Recall: How the model can avoid false negatives.
                F1 Score: How the model can balance precision and recall.
                AUC (Area Under the ROC Curve): How the model can summarize the ROC curve.
            """
        ]
        
        # define elements
        elements =[
            html.H3('Model Description', style={'textAlign': 'center', 'margin': '30px auto'}),
            dbc.Row([
                html.H4("Core Applications", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_ha_cards(ct_titles, ct_descripts, "Ul", 500),
                S_DASH_LINE,
                html.H4("Descriptions of Three Models", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_cards_acccordion(m1_titles, m1_descripts, "Ul"),
                S_DASH_LINE,
                html.H4("Model KPIs / Metrics", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_cards_acccordion(m2_titles, m2_descript, type_='Ul')
            ]),
            DASH_LINE
        ]

        return dbc.Container(elements)

    def model_dataset(self):
        elements = [
            html.H3('Model Dataset Creation', style={'textAlign': 'center', 'margin': '30px auto'}),
            dbc.Accordion([
                dbc.AccordionItem(
                    title='Modifying the Dataset',
                    children=[
                    # Card 1
                    self.design_cards_size(
                        contents=[
                            # Independent Variables Multi Dropdown
                            dbc.Row([
                                html.Label('Select Independent Variables:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                                dcc.Dropdown(
                                    id='independent-variable-dropdown',
                                    options=[
                                        {'label': 'Consumer Sentiment (CSENT)', 'value': 'CSENT'},
                                        {'label': 'Industrial Production (IPM)', 'value': 'IPM'},
                                        {'label': 'New One Family Houses Sold (HOUSE)', 'value': 'HOUSE'},
                                        {'label': 'Unemployment Rate (UNEMP)', 'value': 'UNEMP'},
                                        {'label': 'Long-term Real Interest Rate (LRIR)', 'value': 'LRIR'}
                                    ],
                                    value=['CSENT', 'IPM', 'HOUSE', 'UNEMP', 'LRIR'],  # Default values
                                    placeholder="Select Independent Variables",
                                    multi=True,
                                    style={ 'color': 'gray'}
                                ),
                            ]),
                            # Target & Moving Averages
                            dbc.Row([
                                # Target Variable Dropdown
                                html.Div([
                                    html.Label('Select Target Variable:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                                    dcc.Dropdown(
                                        id='target-variable-dropdown',
                                        options=[
                                            {'label': 'S&P 500 Index', 'value': 'SP500'},
                                            {'label': 'Consumer Sentiment (CSENT)', 'value': 'CSENT'},
                                            {'label': 'Industrial Production in Manufacturing (IPM)', 'value': 'IPM'},
                                            {'label': 'New One Family Houses Sold (HOUSE)', 'value': 'HOUSE'},
                                            {'label': 'Unemployment Rate (UNEMP)', 'value': 'UNEMP'},
                                            {'label': 'Long-term Real Interest Rate (LRIR)', 'value': 'LRIR'}
                                        ],
                                        placeholder="Select a Target Variable",
                                        value='SP500',  # Default value
                                        style={'width': 'fit_content',  'color': 'gray'}
                                    ),
                                ]),
                                # Moving Averages Multi Dropdown
                                html.Div([
                                    html.Label('Select Moving Averages:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                                    dcc.Dropdown(
                                        id='moving-averages-dropdown',
                                        options=[{'label': f'{i} Month(s)', 'value': i} for i in range(1, 7)],
                                        value=[1, 2, 3],  # Default values
                                        placeholder="Select Moving Averages",
                                        multi=True,
                                        className='dropdown',
                                        style={ 'color': 'gray'}
                                    ),
                                ]),
                            ],
                            style={'justify': 'center', 'display': 'inline-flex'}
                            )
                        ]
                    ),
                    # Card 2
                    self.design_cards_size(
                        contents = [
                            # Future Prediction Multi Dropdown
                            dbc.Row([    
                                html.Label('Select Future Prediction Months:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                                dcc.Dropdown(
                                    id='future-prediction-dropdown',
                                    options=[{'label': f'{i} Month(s)', 'value': i} for i in range(1, 7)],
                                    value=list(range(1, 7)),  # Default values
                                    placeholder="Select Future Prediction Months",
                                    multi=True,
                                    className='dropdown',
                                    style={ 'color': 'gray'}
                                ),
                            ]),
                            # Initial Data Size & Poly Degree
                            dbc.Row([    
                                # Initial Training Data Size Input
                                html.Div([
                                    html.Label('Select Initial Training Data:', style={'color': '#d9d9d9', 'paddingBottom': '10px', 'width': '100%'}),
                                    dcc.Input(
                                        id='initial-training-data-size',
                                        type='number',
                                        value=120,
                                        placeholder='Enter initial training data size',
                                        style={'width': 'fit_content',  'marginLeft': '10px'},
                                    ),
                                ]),
                                # Degree of Polynomial
                                html.Div([
                                    html.Label('Select Polynomial Degree:', style={'color': '#d9d9d9', 'paddingBottom': '10px', 'width': '100%'}),
                                    dcc.Input(
                                        id='poly-degree',
                                        type='number',
                                        value=1,
                                        min=1,
                                        max=3,
                                        placeholder='Enter Degree of Polynomials',
                                        style={'width': 'fit_content', 'marginLeft': '10px'}
                                    ),
                                ]),
                            ],
                            style={'justify': 'center'}
                            ),
                        ]
                    ),
                    # Confirmation 
                    dcc.ConfirmDialog(
                        id={'obj': 'confirmation', 'action': 'model-dataset'},
                        message='Are you sure you want to finalize the model dataset?',
                    ),
                    # Finalize Button
                    html.Button('Finalize the Model Dataset', 
                            id={'obj': 'finalize-button', 'action': 'model-dataset'}, 
                            n_clicks=0, 
                            style=BUTTON_STYLE
                    ),    
                ])
            ],
            start_collapsed=True,
            ),

            # Placeholder for Output
            dbc.Row(
                children=[
                    html.H5("Data Preview", style={'color': '#d9d9d9', 'margin': '20px'}),
                    dash_table.DataTable(
                        id={'obj': 'output-table', 'action': 'model-dataset'},
                        data=self.new_df.head().round(2).to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in self.new_df.columns],
                        **TABLE_STYLE,
                    ),
                ]
            ),
            DASH_LINE
        ]

        return dbc.Container(elements)

    def model_result(self):
        elements = [
            html.H3('Model Results / Performances', style={'textAlign': 'center', 'margin': '30px auto'})
        ]

        # Descriptions of Default settings and Hyperparameters
        settings = """
            Independent Variables: 'CSENT', 'IPM', 'HOUSE', 'UNEMP', 'LRIR'
            Target Variables: 'SP500'
            Moving Averages: From One to Three Months
            Future Predictions: From One to Six Months
            Scopes: One, Three, Six, Nine, Twelve Months
        """
        hyperparams = """
            "nta_": Learning rate of the gradient descent (default: 0.01).
            "alpha_": degree of how strong the regularizations are (default: 0.1).
            "lambda_": balancer between l2 and l1 norm (default: 0.5).
            "iter_": iteration of parameter updates at each increment step (default: 100).
            "max_death_": the maximum depth of the tree; only CART model (default: 5).
        """
        elements += [
            # description of default setting and parameters
            self.design_cards_size(
                titles=['Default Model Settings', 'Default Hyperparameters'],
                texts=[settings, hyperparams],
                type_='Ul',
                card_width=[6,6]
                ),
        ]

        # Section of the Customize Model
        model_custom = [

            # Confirmation 
            dcc.ConfirmDialog(
                id={'obj': 'confirmation', 'action': 'model'},
                message='Are you sure you want to start learning new model?',
            ),
            # Finalize Button
            html.Button('Finalize Settings and Launch New Model', 
                    id={'obj': 'finalize-button', 'action': 'model'}, 
                    n_clicks=0, 
                    style=BUTTON_STYLE
            ), 
        ]

        elements += [
            self.design_accordion(titles=['Customize the Models'], contents=[model_custom]),
            S_DASH_LINE,
        ]


        # Model Results (Defaults + Customized)
        # tab titles
        titles = ['Linear Regression', 'Logistic Regression', 'Classification and Regression Tree (CART)']
        
        # Linear Regression
        #  comments / observations
        texts_lin_kpi = """
            example
            example
        """
        texts_lin_be = """
            example
            example
        """
        texts_lin_coef = """
            example
            example
        """
        #  elements for linear 
        linear_elements = [
            html.H4("Regression Metrics / KPIs", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            # regression KPIs
            self.design_ha_figs(figs=[self.fig_data['LinR'][key] for key in ['RMSE', 'SE', 'R2', 'adj_R2']], legends=True),
            self.design_observe(texts=texts_lin_kpi, type_='Ul', title='Observations: '),
            S_DASH_LINE,
            html.H4("Backward Elimination & Development of Coefficients", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            self.design_ha_figs(
                figs=[self.fig_data['LinR']['BE'], self.fig_data['LinR']['Coef']], 
                comments=[texts_lin_be, texts_lin_coef], 
                type_='Ul'
            ),
            S_DASH_LINE,
            html.H4("Prediction Error and Its Distribution", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            self.design_cards_size(
                card_width=[4,4,4],
                contents=[
                    # Moving Averages Select Dropdown
                    html.Div([
                        html.Label('Moving Averages:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                        dmc.Select(
                            id={'model': 'LinR_0', 'obj': 'ma'},
                            placeholder="Select option",
                            data=[{'label': f'{i} Month(s)', 'value': str(i)} for i in range(1, 7)], 
                            value='1',  # Default value
                            required=True,
                            style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}),
                    ]),
                    # Future Prediction Select Dropdown
                    html.Div([
                        html.Label('Predicted Months:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                        dmc.Select(
                            id={'model': 'LinR_0', 'obj': 'fp'},
                            placeholder="Select option",
                            data=[{'label': 'Mean', 'value': 'mean'}] + [{'label': f'{i} Month(s)', 'value': str(i)} for i in range(1, 7)], 
                            value="mean",  # Default value
                            required=True,
                            style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}
                        ),
                    ]),
                    # Scopes Select Dropdown
                    html.Div([
                        html.Label('Scopes:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                        dmc.Select(
                            id={'model': 'LinR_0', 'obj': 'sc'},
                            placeholder="Select option",
                            data=[{'label': 'Mean', 'value': 'mean'}, {'label': f'{1} Month(s)', 'value': '1'}] + [{'label': f'{i} Month(s)', 'value': str(i)} for i in range(3, 13, 3)],  
                            value="mean",  # Default value
                            required=True,
                            style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}
                        )
                    ]),
                ]
            ),
            html.Button(
                'Generate New Figures', 
                id={'obj': 'finalize-button', 'action': 'plot-detail-perf', 'model': 'LinR_0'}, 
                n_clicks=0, 
                style=BUTTON_STYLE
            ),   
            self.design_ha_figs(
                figs=[*self.detail_perf('LinR', self.fig_data['LinR']['pred_df'], 1)],
                model='LinR_0'
            )
        ]

        # Logistic Regression
        #  comments / observations
        texts_log_kpi = """
            example
            example
        """
        texts_log_be = """
            example
            example
        """
        texts_log_coef = """
            example
            example
        """
        #  elements for logistic regression
        logit_elements = [
            html.H4("Classification Metrics / KPIs", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            # classification KPIs
            self.design_ha_figs(figs=[self.fig_data['LogR'][key] for key in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']], legends=True),
            S_DASH_LINE,
            html.H4("Backward Elimination & Development of Coefficients", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            self.design_ha_figs(
                figs=[self.fig_data['LogR']['BE'], self.fig_data['LogR']['Coef']], 
                comments=[texts_log_be, texts_log_coef],
                type_='Ul'
            )
        ]

        # CART
        #  comments / observations
        texts_cart_kpi = """
            example
            example
        """
        texts_cart_coef = """
            example
            example
        """
        cart_elements = [
            html.H4("Regression Metrics / KPIs", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            # regression KPIs
            self.design_ha_figs(figs=[self.fig_data['CART'][key] for key in ['RMSE', 'SE', 'R2', 'adj_R2']], legends=True),
            S_DASH_LINE,
            html.H4("Development of Coefficients", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            self.design_ha_figs(
                figs=[self.fig_data['CART']['Coef']], 
                comments=[texts_cart_coef],
                type_='Ul'
            ),
            html.H4("Prediction Error and Its Distribution", style={'color': '#d9d9d9', 'margin': '20px 0'}),
            self.design_cards_size(
                card_width=[4,4,4],
                contents=[
                    # Moving Averages Select Dropdown
                    html.Div([
                        html.Label('Moving Averages:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                        dmc.Select(
                            id={'model': 'CART_0', 'obj': 'ma'},
                            placeholder="Select option",
                            data=[{'label': f'{i} Month(s)', 'value': str(i)} for i in range(1, 7)], 
                            value='1',  # Default value
                            required=True,
                            style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'}),
                    ]),
                    # Future Prediction Select Dropdown
                    html.Div([
                        html.Label('Predicted Months:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                        dmc.Select(
                            id={'model': 'CART_0', 'obj': 'fp'},
                            placeholder="Select option",
                            data=[{'label': 'Mean', 'value': 'mean'}] + [{'label': f'{i} Month(s)', 'value': str(i)} for i in range(1, 7)], 
                            value="mean",  # Default value
                            required=True,
                            style={'backgroundColor': '#333', 'color': 'white', 'borderColor': '#555'},
                        ),
                    ]),
                    # Scopes Select Dropdown
                    html.Div([
                        html.Label('Scopes:', style={'color': '#d9d9d9', 'paddingBottom': '10px'}),
                        dmc.Select(
                            id={'model': 'CART_0', 'obj': 'sc'},
                            placeholder="Not Options",
                            value="mean",
                            disabled=True,
                        )
                    ]),
                ]
            ),
            html.Button(
                'Generate New Figures', 
                id={'obj': 'finalize-button', 'action': 'plot-detail-perf', 'model': 'CART_0'}, 
                n_clicks=0, 
                style=BUTTON_STYLE
            ),   
            self.design_ha_figs(
                figs=[*self.detail_perf('CART', self.fig_data['CART']['pred_df'], 1)],
                model='CART_0'
            )
        ]

        # Aggregate all model elements
        elements += [
            html.H4('Visualizations of Each Model', style={'color': '#d9d9d9', 'margin': '40px 10px'}),
            # summary of each model result
            dbc.Tabs(
                id = 'model-results-tab',
                children=[
                    dbc.Tab(
                        label=title, 
                        children=[html.Div(content)],
                    )
                    for title, content in zip(titles, [linear_elements, logit_elements, cart_elements])
                ]
            ),
            DASH_LINE
        ]

        return dbc.Container(elements)



    def model_finalize(self):
        pass
    
    def conclution(self):
        pass

    def further_approach(self):
        pass

    # Plotly Figure
    def detail_perf(self, model: str, pred_df: pd.DataFrame, ma: int, fp: int|str = 'mean', sc: int|str = 'mean'):
        """
            Return two plotly chart:
            - Line chart: Comparing the predicted and actual value.
            - Mix chart:  Distribution of the predicted errors by histogram and scatter plots.

            Return: plotly.graph_objs._figure.Figure
        """
        # modify model name if needed
        if '_' in model:
            model = model.split('_')[0]

        # when model is CART, sc is not optional
        if model == 'CART':
            sc = 'mean'

        # index as Date
        pred_df = pred_df.set_index('Date')
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
            sub_title1 = f'<br><span {SUB_CSS}> --Adjusted parameters to predict a target price {fp} month(s) ahead.</span>'
            sub_title2 = f'<br><span {SUB_CSS}> --Focused on {sc} month(s) of data to update parameters at each step.</span>'
            draw_bands = False
        #  if only sc is 'mean
        elif fp != 'mean' and sc == 'mean':
            filter_cols = [c for c in pred_df.columns if c.startswith(f'Pred_{ma}MA_{fp}FP')]
            predict = pred_df[filter_cols].mean(axis=1)
            sub_title1 = f'<br><span {SUB_CSS}> --Adjusted parameters to predict a target price {fp} month(s) ahead.</span>'
            sub_title2 = f'<br><span {SUB_CSS}> --Aggregated different prediction results generated from all scopes.</span>'
            if model == 'CART':
                draw_bands = False
        #  if only fp is 'mean'
        elif fp == 'mean' and sc != 'mean':
            filter_cols = [c for c in pred_df.columns 
                        if c.startswith(f'Pred_{ma}MA') and c.endswith(f'_{sc}SC')]
            predict = pred_df[filter_cols].mean(axis=1)
            sub_title1 = f'<br><span {SUB_CSS}> --Aggregated multiple results generated from all defined future months ahead.</span>'
            sub_title2 = f'<br><span {SUB_CSS}> --Focused on the recent {sc} month(s) of data to update parameters at each step.</span>'
        #  else; both are 'mean'
        else:
            filter_cols = [c for c in pred_df.columns if c.startswith(f'Pred_{ma}MA')]
            predict = pred_df[filter_cols].mean(axis=1)
            sub_title1 = f'<br><span {SUB_CSS}> --Aggregated all results from different futures ahead & scopes.</span>'
            sub_title2 = f'<br><span {SUB_CSS}> --Bands showing three standard deviations based on all results. </span>'

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
        sub = f"<br><span {SUB_CSS}> -- Histogram shows the frequency of each predicted error.</span>"
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

    # Design Shortcut
    def design_texts(self, texts: str, type_: str = "P"):
        # define style
        content_style={'lineHeight': 2, 'color': 'F4F4F4'}
        # define list or paragraph
        if type_ == 'Ul':
            lines = texts.strip().split('\n')
            return html.Ul([html.Li(c, style={'margin': '5px 0'}) for c in lines], style=content_style)
        
        elif type == 'Ol':
            lines = texts.strip().split('\n')
            return html.Ol([html.Li(c, style={'margin': '5px 0'}) for c in lines], style=content_style)
        
        else:
            return html.P(texts, style=content_style)

    def design_observe(self, texts: str, type_: str = 'P', title: str = 'Graph Interpretation'):
        div = html.Div([
            html.H5(title, style={'color': '#d9d9d9', 'padding': '0 0 10px'}),
            self.design_texts(texts, type_),
        ], style={
            'padding': '20px',
            'border': '1px solid #444',
            'borderRadius': '5px',
            'backgroundColor': '#2c2c2c',
            'margin': '10px',
            'lineheight': 2
        })

        return div

    def design_tabs(self, titles: list, contents: list, **styles):    
        elements = html.Div([
            dbc.Tabs([
                dbc.Tab(
                    label=title, 
                    children=[html.Div(content)],
                )
                for title, content in zip(titles, contents)
            ])
        ],
        style={**styles}
        )

        return elements

    def design_accordion(self, titles: list, contents: list):
        elements = dmc.AccordionMultiple([
            dmc.AccordionItem([
                dmc.AccordionControl(title, style={'backgroundColor': '#444', 'color': '#F9F9F9', 'margin': '5px 0'}),
                dmc.AccordionPanel(children=content, style={'backgroundColor': '#333'}),
            ],
            value=title,
            )
            for title, content in zip(titles, contents)
        ],
        variant='filled',
        style={'margin': '30px 20px'}
        )
        
        return elements

    def design_cards_size(self, titles: list = [], texts: list = [], type_: str = 'P', contents: list = [], card_width: list = [6,6]):
        """
        A single card; two sections for two contents
        """
        col_sizes = len(titles) or len(texts) or len(contents)

        element = dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        # titles and texts
                        html.Div([
                            html.H5(titles[i], className="card-title", style={'padding': '0 0 10px'}),
                            self.design_texts(texts[i], type_)
                        ]) if titles and texts else None,
                        # returrn each content
                        contents[i] if contents else None
                        ],
                        style={"padding": "20px", 
                               "borderRight": "2px dashed #777777" if i != col_sizes-1 else 'None'},
                        width=card_width[i]
                    )
                    for i in range(col_sizes)
                ]),
            ]),
        )

        return element

    def design_cards_equal(self, titles: list, contents: list, type_: str = 'P', mode: str = 'ha'):
        # direction
        f_direct = 'row' if mode == 'ha' else 'column'
        # define elements
        elements = html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5(title, className='card-title', style={'color': 'F9F9F9', 'margin': '5px 0'})),
                            dbc.CardBody(
                                [self.design_texts(content, type_)], 
                                style={'height': '100%', 'fontSize': 16})
                        ], 
                        style={'height': '100%', 'margin': 'auto', 'marginBottom': '15px'}
                    ), 
                    style={'margin': '10px'}
                )
                for title, content in zip(titles, contents) 
            ], style={'flexDirection': f_direct, 'justifyContent': 'space-around', 'width': '100%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'})

        return elements

    def design_cards_acccordion(self, titles: list, contents: list, type_: str = 'P', mode: str = 'ha'):
        # direction
        f_direct = 'row' if mode == 'ha' else 'column'
        # define elements
        elements = html.Div([
            dbc.Row([
                # multiple cols
                dbc.Col(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                title=title,
                                children=dbc.Card(
                                    dbc.CardBody(
                                        self.design_texts(content, type_),
                                        style={'height': '100%', 'fontSize': '16px'}
                                    ),
                                    style={'height': '100%', 'margin': 'auto'}
                                ),
                                style={'fonrSize': '20px', 'color': '#F9F9F9', 'margin': '5px'}
                            )
                        ],
                        start_collapsed=False,
                    ),
                    style={'margin': '10px'}
                )
                for title, content in zip(titles, contents) 

            ], style={'flexDirection': f_direct, 'justifyContent': 'space-around', 'width': '100%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'})

        return elements 

    def design_ha_cards(self, titles: list, contents: list, type_: str = 'P', width: int = 400):
        elements = [
            dbc.Row(
                # multiple cols
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H5(title, className='card-title', style={'color': 'F9F9F9', 'margin': '5px 0'})
                                ),
                                dbc.CardBody(
                                    [self.design_texts(content, type_)], 
                                    style={'height': '100%', 'fontSize': 16})
                            ], 
                            outline=True,
                            style={'height': '100%', 'margin': 'auto', 'marginBottom': '15px'},
            
                        ),
                        className='flex-nowrap',
                        style={'display': 'inline-flex', 'minWidth': width,  'maxWidth': width, 'margin': '10px'}
                    ) for title, content in zip(titles, contents)
                ],
                style={'display': 'inline-flex', 'flex-wrap': 'nowrap', 'overflowX': 'auto', 'width': '100%', 'gap': '10px'},
            )
        ]

        return dbc.Row(elements, style={'margin': '15px auto'})

    def design_ha_figs(self, figs: list, model: str = '', legends: bool = False, comments: list = [], type_='P'):
        # define return
        elements = []
        # update group id number
        self.design_ha_figs_id += 1
        # define base (default) fig id
        base_fig_id = {'g_id': self.design_ha_figs_id}
        # define default display format for comment
        c_display = 'block'

        # if model name is assigned
        if model:
            base_fig_id = {'model': model}
        
        # commom legends
        if legends:
            # decide base id pf each figure
            base_fig_id = {'func': 'ha_figs_', 'obj': 'fig', 'g_id': self.design_ha_figs_id}
            # legend options (assume every figure has same legend)
            options = [getattr(fig, 'legendgroup', str(i)) for i, fig in enumerate(figs[0].data)]
            # color
            colors = [FIG_COLOR(fig) for fig in figs[0].data]

            elements += [
                dmc.ChipGroup(
                    id={'func': 'ha_figs_', 'obj': 'legend', 'g_id': self.design_ha_figs_id},
                    children=[
                        dmc.Chip(
                            children=str(opt),
                            value=str(opt),
                            variant="outline", 
                            color=color
                            ) for opt, color in zip(options, colors)
                    ],
                    value=[str(opt) for opt in options],
                    multiple=True,
                    style={'display': 'flex', 'justifyContent': 'center'}
                )
            ]

        # define comments
        if not comments:
            c_display = 'None'
            # create empty list
            comments = [''] * len(figs)

        # add main elements
        elements += [
            dbc.Row(
                [dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody([
                                dcc.Graph(
                                    id={**base_fig_id, 'id': str(i)},
                                    figure=fig,
                                ),
                                dbc.CardFooter(
                                    self.design_texts(texts=comments[i], type_=type_), 
                                    style={'display': c_display, 'padding': '20px'}
                                ),
                            ], style={'margin': '5px', 'maxWidth': 'min-content'}),
                        ],
                        outline=True,
                    ),
                    className='flex-nowrap',
                    style={'display': 'inline-flex', 'minWidth': 'fit-content', 'margin': '10px'}
                ) for i, fig in enumerate(figs)],
                style={'display': 'inline-flex', 'flex-wrap': 'nowrap', 'overflowX': 'auto', 'width': '100%', 'gap': '10px'},
            )
        ]

        return dbc.Row(elements, style={'margin': '15px auto'})

    def callbacks(self):

        # Matching legend in horizontal (ha) figure plots
        @self.app.callback(
            output=Output({'func': 'ha_figs_', 'obj': 'fig', 'g_id': MATCH, 'id': ALL}, 'figure'),
            inputs=Input({'func': 'ha_figs_', 'obj': 'legend', 'g_id': MATCH}, 'value'),
            state=State({'func': 'ha_figs_', 'obj': 'fig', 'g_id': MATCH, 'id': ALL}, 'figure'),
            prevent_initial_call=True
        )
        def update_visibility(value, fig):
            # Determine which input was triggered
            triggered_id = ctx.triggered_id
            if triggered_id:
                # define output
                outputs = []
                # get the checked value as set 
                checked = set(value)
                # traversing all figure data
                for f in fig:
                    # define patch
                    p = Patch()
                    for i in range(len(f['data'])):
                        p['data'][i].update({'visible': f['data'][i]['name'] in checked})
                    
                    outputs.append(p)
                
                return outputs
            
            else:
                return no_update

        # Confirmation to a certain action
        @self.app.callback(
            Output({'obj': 'confirmation', 'action': MATCH}, 'displayed'),
            [Input({'obj': 'finalize-button', 'action': MATCH}, 'n_clicks')],
            prevent_initial_call=True,
        )
        def display_confirm_dialog(n_clicks):
            if n_clicks > 0:
                return True
            return False

        # Finalizing model dataset creation
        @self.app.callback(
            [Output({'obj':'output-table', 'action': 'model-dataset'}, 'data'),
             Output({'obj':'output-table', 'action': 'model-dataset'}, 'columns')],
            [Input({'obj':'confirmation', 'action': 'model-dataset'}, 'submit_n_clicks')],
            [State('independent-variable-dropdown', 'value'),
            State('target-variable-dropdown', 'value'),
            State('moving-averages-dropdown', 'value'),
            State('future-prediction-dropdown', 'value'),
            State('initial-training-data-size', 'value'),
            State('poly-degree', 'value')],
            prevent_initial_call=True
        )
        def model_data_creation(submit_n_clicks, independent_vars, target_var, ma, fp, init_data, poly_degree):
            if submit_n_clicks:
                if not all([independent_vars, target_var, ma, fp, init_data, poly_degree]):
                    return no_update

                # update self.PA
                self.new_df = self.PA.create_data(
                    X_n=independent_vars, y_n=target_var,
                    ma=ma, fp=fp, init_train=init_data, poly_d=poly_degree
                )
                data = self.new_df.head().round(2).to_dict('records')
                cols = [{'name': i, 'id': i} for i in self.new_df.columns]
                
                return data, cols
            

        # Launch the customized model
        """@self.app.callback(
            Output('model-results-tab', 'children'),
            Input({'obj':'confirmation', 'action': 'model'}, 'submit_n_clicks'),
            #[
            #    State(),
            #    State(),
            #]
            prevent_initial_call=True
        )
        def custom_model_result(submit_n_clicks):
            if submit_n_clicks:
                # make sure all state variables are not blank
                if not all([...]): # all
                    return no_update
                # use Patch
                p = Patch()
                # list for new results
                new_results = []
                # train model
                

                # append all results
                p.append(new_results)

                return p
        """

        # Update Detailed Performance
        @self.app.callback(
            output=dict(
                line=Output({'model': MATCH, 'id': '0'}, 'figure'),
                dist=Output({'model': MATCH, 'id': '1'}, 'figure'),
            ),
            inputs=dict(
                clicks=Input({'obj': 'finalize-button', 'action': 'plot-detail-perf', 'model': MATCH}, 'n_clicks')
            ),
            state=dict(
                ma=State({'model': MATCH, 'obj': 'ma'}, 'value'),
                fp=State({'model': MATCH, 'obj': 'fp'}, 'value'),
                sc=State({'model': MATCH, 'obj': 'sc'}, 'value'),
            ),
            prevent_initial_call=True
        )
        def update_detail_perf(clicks, ma, fp, sc):
            # get triggered id
            trig_id = ctx.triggered_id
            if clicks > 0 and trig_id:
                # get the triggered model
                model = trig_id['model'].split('_')[0]
                # generate new figure
                line_, dist_ = self.detail_perf(model, self.fig_data[model]['pred_df'], ma, fp, sc)

                return dict(line=line_, dist=dist_)

            else:
                return no_update