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
], fluid=True)

# define all callbacks
design.callbacks()


if __name__ == "__main__":
    app.run_server(debug=True)
