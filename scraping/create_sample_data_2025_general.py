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
                "Eric L. Adams": np.round(np.random.rand(len(elect_dists)) * 5),
                "Curtis A. Sliwa": np.round(np.random.rand(len(elect_dists)) * 50),
                "Andrew M. Cuomo": np.round(np.random.rand(len(elect_dists)) * 100),
                "Irene Estrada": np.round(np.random.rand(len(elect_dists)) * 5),
                "Zohran Kwame Mamdani": np.round(np.random.rand(len(elect_dists)) * 150),
                "Joseph Hernandez": np.round(np.random.rand(len(elect_dists)) * 5),
                "Jim Walden": np.round(np.random.rand(len(elect_dists)) * 5),
                "WRITE-IN": np.round(np.random.rand(len(elect_dists)) * 5),
            }
        ).assign(
            AD=lambda df: df.ElectDist.str[:2].astype(int),
            ED=lambda df: df.ElectDist.str[2:].astype(int),
        )
    ),
    open("data/cache/Mayor (FAKE DATA).json", "w"),
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
                "Luis E. Quero": np.round(np.random.rand(len(subset_elect_dists)) * 100),
                "WRITE-IN": np.round(np.random.rand(len(subset_elect_dists)) * 5),
            }
        ).assign(
            AD=lambda df: df.ElectDist.str[:2].astype(int),
            ED=lambda df: df.ElectDist.str[2:].astype(int),
        )
    ),
    open("data/cache/Member of the City Council 38th Council District (FAKE DATA).json", "w"),
    indent=4,
)
