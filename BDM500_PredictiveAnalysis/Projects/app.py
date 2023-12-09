from dash import Dash
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

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

design = Design(app, origin_df)

app.layout = dbc.Container([
    design.header(),
    #design.overview(),
    #design.dataset(),
    #design.data_preprocessing(),
    #design.data_observation(),
    design.model_descript(),
    #design.model_dataset(),
    design.model_result()
], fluid=True)

# define all css styles and callbacks
"""
.Select-value-label,
.Select-value-icon
{
    color: #cce5ff !important;
}
"""
#design.callbacks()

if __name__ == "__main__":
    app.run_server(debug=True)
