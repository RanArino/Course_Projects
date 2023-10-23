from dash import Dash, dash_table
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd

import app_design

# import the dataset from csv file
df = pd.read_csv('original.csv')

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    app_design.header(), 
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True)
