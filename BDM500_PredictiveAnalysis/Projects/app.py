from dash import Dash, dash_table
import dash_bootstrap_components as dbc

import os
import numpy as np
import pandas as pd

import app_design

# current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the file path
csv_file = os.path.join(current_dir, "original.csv")
# load csv
df = pd.read_csv(csv_file)

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    app_design.header(), 
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True)
