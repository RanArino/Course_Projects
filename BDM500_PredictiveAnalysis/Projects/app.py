#####
# This file is main python file for web app
#####
from dash import Dash
from flask import Flask
import dash_bootstrap_components as dbc

import os
import numpy as np
import pandas as pd

from app_design import Design

# current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the file path
csv_file = os.path.join(current_dir, "original.csv")
# load csv
origin_df = pd.read_csv(csv_file)

app = Dash(external_stylesheets=[
    dbc.themes.DARKLY,
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
    ])
server: Flask = app.server  # type: ignore

design = Design(app, origin_df, current_dir)

app.layout = dbc.Container([
    design.header(),
    design.overview(),
    design.dataset(),
    design.data_preprocessing(),
    design.data_observation(),
    design.model_descript(),
    design.model_dataset(),
    design.model_result(),
    design.model_finalize(),
    design.conclution(),
    design.further_approach(),
], fluid=True, style={'padding': '0'})

# define all callbacks
design.callbacks()

@server.route('/health')
def health_check():
    return 'Healthy', 200 

if __name__ == "__main__":
    app.run_server(debug=True)