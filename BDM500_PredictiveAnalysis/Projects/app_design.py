from dash import html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc


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
                                #dbc.Col(html.Img(src="/assets/logo.png", height="60px")),
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
    project_summary = """
    This project aims to build predictive models for the S&P500 index using macroeconomic indicators from the Federal Reserve Economic Data (FRED). 
    The goal is to predict the performance of the S&P500 index for the upcoming one, two, and three months. 
    The models will be assessed under different conditions such as various moving averages, different learning mehods, and types of machine learning models. 
    The project will also identify which economic indicators significantly affect the S&P500 index's future performance. 
    The intended audience includes stakeholders in finance, economics, and stock market fields who require accurate predictions for their investment strategies.
    """
    summary_parts = project_summary.split('. ')

    icons = ['chart-line', 'calendar-alt', 'balance-scale', 'search-dollar', 'users']

    cards = html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4('Section {}'.format(i+1), className='card-title')),
                dbc.CardBody([
                    html.P(part),
                    html.I(className='fas fa-{}'.format(icons[i]), style={'fontSize': '24px'})
                ], style={'height': '100%'})
            ], style={'height': '100%'}), style={'margin': '10px'}) for i, part in enumerate(summary_parts)
        ], style={'justifyContent': 'around', 'marginRight': 0, 'marginLeft': 0})
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'})
    
    return html.Div([
        html.H3('Project Overview', style={'textAlign': 'center'}),
        cards
    ])