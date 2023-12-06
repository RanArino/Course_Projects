from dash import html, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import pandas as pd


def header():
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

def overview():
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
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H4(title[i], className='card-title', style={'color': 'F9F9F9'})),
                                dbc.CardBody(
                                    [html.P(content[i], style={'lineHeight': 2, 'color': 'F4F4F4'})], 
                                    style={'height': '100%', 'fontSize': 17})
                            ], 
                            style={'height': '100%', 'margin': 'auto'}
                        ), 
                        style={'margin': '10px'}
                    )
                    for i in range(len(title)) 
                ], style={'justifyContent': 'around', 'marginRight': 0, 'marginLeft': 0, 'width': '100%'}
                ),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justify_content': 'center'}),
            html.Hr(style={'borderTop': '2px dashed #fff', 'margin': '75px 0'}),
        ])
    for main, title, content in zip(main_titles_lst, titles_lst, contents_lst)
    ]
    
    return dbc.Container(cards)

def dataset():
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

    table_style=dict(
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

    table = [
        html.H3("Dataset Information", style={'textAlign': 'center', 'margin': '30px auto'}),
        # Table
        dash_table.DataTable(
            id='table',
            columns=[
                {"name": i, "id": i} if i != 'Data Source' else {"name": i, "id": i, "type": "text", "presentation": "markdown"} for i in df.columns
            ],
            data=df.to_dict('records'),
            **table_style
        ),
        # Notifications section
        html.Div(
            [html.P(note, style={'color': '#eeeeee', 'fontSize': '0.85em'}) for note in notifications],
        style={'backgroundColor': '#333333', 'padding': '15px', 'margin': '20px 0', 'borderRadius': '5px'}
        ),
    ]

    return dbc.Container(table)

def data_preprocessing(df: pd.DataFrame):
    """
    Showing the data preprocessing phases step by step.
    """