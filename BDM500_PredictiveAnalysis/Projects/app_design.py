from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px

import pandas as pd

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
#  plotly figure design
FIGURE_STYLE = dict(
    template='plotly_dark',
)
# horizontal dash line
DASH_LINE = html.Hr(style={'borderTop': '2px dashed #fff', 'margin': '75px 0'})


class Design:
    def __init__(self, df: pd.DataFrame):
        # assign original data frame
        self.origin = df

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
        self.corr1 = self.df3[['MY10Y', 'CPI', 'CSENT', 'IPM', 'HOUSE', 'UNEMP', 'SP500']].corr().reset_index(names='')
        # add new data
        self.df3.insert(loc=9, column='LRIR', value=self.df3['MY10Y'] - self.df3['CPI'])
        # drop MY10Y and CPI
        self.df3.drop(['MY10Y', 'CPI'], axis=1, inplace=True)
        # show new correlation matrix
        self.corr2 = self.df3[['CSENT', 'IPM', 'HOUSE', 'UNEMP', 'LRIR', 'SP500']].corr().reset_index(names='')


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
        main_titles_lst = [
            'Project Overview',
            'Objectives / Questions',
            'Expected Concerns'
        ]
        # titles
        titles_lst = [
            ['Main Theme', 'Intended Audiences', 'Applications'],
            ['Various Aspects', 'Facter Analysis', 'Model Performance'],
            ['Distribution of Data', 'Lag of Data Release', 'Market Volatility', 'Complex Relationship']
        ]
        # contents
        contents_lst = [
            [
                'Building incrementally-learning predictive models for the S&P500 index using macroeconomic indicators.',
                'Stakeholders in finance, economics, and stock market, particularly in mutual funds, investment banks, and pension funds​.',
                'Guiding investment decisions, Informing risk management strategies, Analysis for traders and institutional investors',
            ],
            [
                'How will the future performance of the S&P 500 index be changed by different conditions; moving averages, scopes, future predictions, and different models?',
                'Which economic indicators are likely to significantly affect the future performance of the S&P500 index, and how those impacts have been changed over time?',
                'How accurately can each model predict the performance of the S&P500 index from one to six months ahead based on the latest economic indicator?'
            ],
            [
                html.Ol([
                    html.Li("Economic indicators often do not follow a normal distribution."),
                    html.Li("For example, the unemployment rate tends to be right-skewed, even after transformations"),
                ]),
                html.Ol([
                    html.Li("Economic data is reported with a time lag of about 1 month"),
                    html.Li("Real-time data, like S&P500, differs from economic data, causing timing challenges."),
                ]),
                html.Ol([
                    html.Li("Human emotions impact market decisions, leading to unpredictable market behaviors."),
                    html.Li("Market volatility and randomness pose challenges in explaining market actions."),
                ]),
                html.Ol([
                    html.Li("The market is influenced by complex economic dependencies, disruptions, and geopolitical tensions"),
                    html.Li("These factors hinder the development of accurate predictive models"),
                ]),
            ]

            ]

        # all cards
        cards = [
            html.Div([
                html.H3(main, style={'textAlign': 'center', 'margin': '30px auto'}),
                self.design_cards(titles, contents),
                #html.Div([    
                #], style={'display': 'flex', 'flexWrap': 'wrap', 'justify_content': 'center'}),
                DASH_LINE,
            ])
        for main, titles, contents in zip(main_titles_lst, titles_lst, contents_lst)
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
            {"Symbol": "LRIR", "Dtype": "float", "Description": "Long-term Real Interest rate; the subtraction of MY10Y by %YoY CPI", "Data Source": ""}
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
        (2): 
        """
        elements = [
            html.H3('Data Preprocessing', style={'textAlign': 'center', 'margin': '30px auto'})
        ]

        # (1) Dealing with Incomplete Data
        #  graph
        fig = px.line(self.group_year, x='Year', y='Month', title='Number of monthly data on each year')
        fig.update_layout(
            hovermode="x unified", 
            **FIGURE_STYLE
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
                        self.design_oberve(comments_1, 'Ul'),
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
                DASH_LINE
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
                    self.design_cards(titles, contents),
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

        elements = [html.H3('Data Observations', style={'textAlign': 'center', 'margin': '30px auto'})]

        # (1) Correlation Matrix & Feature Selection
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
        comments_1 = """
            CSENT, IPM, and HOUSE are positively correlated with the S&P500 index, and those relationships are relative high.
            MY10Y, CPI, and UNEMP have weaker relationship with the S&P500 index; all values are close to zero.
            MY10Y and CPI shows the strongest correlation; although there is less risk of multicollinearity, a derived data "LRIR" will be created for feature removal.
        """ 
        comments_2 = """
            Symbol: LRIR
            Name: Longer-term Real Interest Rate
            Calculation: MY10Y - CPI(YoY growth)
            Description: Assuming that the nominal rate can be reflected from MY10Y, and the inflation rate is the YoY percent change in CPI.
        """
        # output elements
        elements += [
            dbc.Row([
                html.H4("Correlation Matrix & Feature Selection", style={'color': '#d9d9d9', 'margin': '20px 0'}),
                self.design_tabs(tab_titles, tab_elements, margin='0 0 20px', ),
                dbc.Row([
                    dbc.Col(self.design_oberve(comments_1, type_='Ul', title='Onservations'), width=6),
                    dbc.Col(self.design_oberve(comments_2, type_='Ul', title='New Derived Data'), width=6),
                ])
            ])
        ]

        return dbc.Container(elements)


    # degisn shortcut
    def design_oberve(self, comments: str, type_: str = '', title: str = 'Graph Interpretation'):
        if type_ == 'Ul':
            lines = comments.strip().split('\n')
            obs_comp = html.Ul([html.Li(c, style={'padding': '5px 0'}) for c in lines], style={'color': '#d9d9d9'})
        elif type_ == 'Ol':
            lines = comments.strip().split('\n')
            obs_comp = html.Ol([html.Li(c, style={'padding': '5px 0'}) for c in lines], style={'color': '#d9d9d9'})
        else:
            obs_comp = html.P(comments, style={'color': '#d9d9d9'})

        div = html.Div([
            html.H5(title, style={'color': '#d9d9d9'}),
            obs_comp,
        ], style={
            'padding': '20px',
            'border': '1px solid #444',
            'borderRadius': '5px',
            'backgroundColor': '#2c2c2c',
            'margin': '10px',
            'lineheight': 2
        })

        return div

    def design_cards(self, title: list, content: list, mode: str = 'ha'):
        f_direct = 'row' if mode == 'ha' else 'column'
        cards = html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(html.H5(title[i], className='card-title', style={'color': 'F9F9F9'})),
                            dbc.CardBody(
                                [html.P(content[i], style={'lineHeight': 2, 'color': 'F4F4F4'})], 
                                style={'height': '100%', 'fontSize': 16})
                        ], 
                        style={'height': '100%', 'margin': 'auto', 'marginBottom': '15px'}
                    ), 
                    style={'margin': '10px'}
                )
                for i in range(len(title)) 
            ], style={'flexDirection': f_direct, 'justifyContent': 'space-around', 'width': '100%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'})

        return cards
    
    def design_tabs(self, titles: list, elements: list, **styles):    
        tabs = html.Div([
            dbc.Tabs([
                dbc.Tab(
                    label=titles[i], 
                    children=[html.Div(elements[i])],
                )
                for i in range(len(elements))
            ])
        ],
        style={**styles}
        )

        return tabs
