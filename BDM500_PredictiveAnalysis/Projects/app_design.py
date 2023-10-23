from dash import html
import dash_bootstrap_components as dbc


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
        style={"height": "100px"},
    )

    return header