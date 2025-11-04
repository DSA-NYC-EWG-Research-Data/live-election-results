import datetime
from datetime import timezone
import json
import urllib
import numpy as np
import pandas as pd
from utils import *

# load districts
elect_dists = json.load(open("data/shapes/districts.json", "r")).keys()

# mayoral election
json.dump(
    get_grouped_dict(
        pd.DataFrame(
            {
                "ElectDist": elect_dists,
                "Reported %": np.random.rand(len(elect_dists)),
                "Adrienne Adams": np.round(np.random.rand(len(elect_dists)) * 50),
                "Andrew Cuomo": np.round(np.random.rand(len(elect_dists)) * 100),
                "Zohran Kwame Mamdani": np.round(np.random.rand(len(elect_dists)) * 100),
                "Selma Bartholomew": np.round(np.random.rand(len(elect_dists)) * 20),
                "Michael Blake": np.round(np.random.rand(len(elect_dists)) * 10),
                "Brad Lander": np.round(np.random.rand(len(elect_dists)) * 9),
                "Zellnor Myrie": np.round(np.random.rand(len(elect_dists)) * 8),
                "Paperboy Price": np.round(np.random.rand(len(elect_dists)) * 8),
                "Jessica Ramos": np.round(np.random.rand(len(elect_dists)) * 8),
                "Scott Stringer": np.round(np.random.rand(len(elect_dists)) * 8),
                "Whitney Tilson": np.round(np.random.rand(len(elect_dists)) * 8),
                "WRITE-IN": np.round(np.random.rand(len(elect_dists)) * 5),
            }
        ).assign(
            AD=lambda df: df.ElectDist.str[:2].astype(int),
            ED=lambda df: df.ElectDist.str[2:].astype(int),
        )
    ),
    open("data/cache/Mayoral (FAKE DATA).json", "w"),
    indent=4,
)

# council election
subset_elect_dists = [elect_dist for elect_dist in elect_dists if elect_dist[:2] in ["61", "51", "49"]]
json.dump(
    get_grouped_dict(
        pd.DataFrame(
            {
                "ElectDist": subset_elect_dists,
                "Reported %": np.random.rand(len(subset_elect_dists)),
                "Alexa Avilés": np.round(np.random.rand(len(subset_elect_dists)) * 200),
                "Ling Ye": np.round(np.random.rand(len(subset_elect_dists)) * 150),
                "WRITE-IN": np.round(np.random.rand(len(subset_elect_dists)) * 5),
            }
        ).assign(
            AD=lambda df: df.ElectDist.str[:2].astype(int),
            ED=lambda df: df.ElectDist.str[2:].astype(int),
        )
    ),
    open("data/cache/City Council 38 (FAKE DATA).json", "w"),
    indent=4,
)
