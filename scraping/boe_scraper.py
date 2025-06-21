import time
import logging
from utils import get_elections, get_per_ed_results, get_election_results
import json
import sys
import argparse
import boto3
import tempfile
import os
import pandas as pd
import numpy as np

# dsa races we care about.
BOE_ELECTION_WHITELIST = [
    {"position": "Mayor", "party": "Democratic"},  # Zohran Mamdani
    {
        "position": "Member of the City Council",
        "party": "Democratic",
        "sub_position": "Member of the City Council 38th Council District",
    },  # Alexa Avilés
]


def setup_logger():
    logger = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.handlers = []
    LOGGER.addHandler(handler)
    return logger


LOGGER = setup_logger()


def gen_dummy_data():
    elect_dists = json.load(open("data/shapes/districts.json", "r")).keys()
    results_dict = get_grouped_dict(
        pd.DataFrame(
            {
                "ElectDist": elect_dists,
                "Reported %": np.random.rand(len(elect_dists)),
                "Nicola (Democratic)": np.round(np.random.rand(len(elect_dists)) * 100),
                "Jeremy (Democratic)": np.round(np.random.rand(len(elect_dists)) * 80),
                "Chris (Republican >:) )": np.round(np.random.rand(len(elect_dists)) * 60),
                "WRITE-IN": np.round(np.random.rand(len(elect_dists)) * 5),
            }
        ).assign(
            AD=lambda df: df.ElectDist.str[:2].astype(int),
            ED=lambda df: df.ElectDist.str[2:].astype(int),
        )
    )
    results_dict["last_updated"] = str(pd.Timestamp.now())
    return results_dict


def fetch_data(args):
    if args.output.startswith("s3:"):
        # s3 output
        bucket = args.output[3:]
        s3 = boto3.client("s3")

        def write_output(data, election_name):
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=True) as tmp:
                json.dump(data, tmp, indent=4)
                tmp.flush()
                s3.upload_file(
                    tmp.name,
                    bucket,
                    f"results/{election_name}.json",
                    ExtraArgs={
                        "ACL": "public-read",
                        "CacheControl": "no-cache, no-store, must-revalidate",
                    },
                )
            # output is mainly for logging not any specific kind of syntax.
            return f"s3:{bucket}/results/{election_name}.json"

    else:
        # local file output
        def write_output(data, election_name):
            fname = os.path.join(args.output, f"results/{election_name}.json")
            with open(fname, "w") as f:
                json.dump(data, f, indent=4)
            return fname

    try:
        elections_dict = get_elections(args.url, LOGGER, whitelist=BOE_ELECTION_WHITELIST)
        LOGGER.info(f"Success. elections_dict:\n{json.dumps(elections_dict, indent=4)}")
    except Exception as e:
        LOGGER.error(f"Error fetching data on get_elections({args.url}):\n{e}")
        return False
    for election, link in elections_dict.items():
        try:
            if args.dummy_data:
                results_dict = gen_dummy_data()
                election += " (FAKE DATA)"
            else:
                results_dict = get_election_results(link, LOGGER)
            fname = write_output(results_dict, election)
            LOGGER.info(f"Sucess. Stored data in: {fname}")
        except Exception as e:
            LOGGER.error(f"Error fetching data on get_per_ed_results({link}):\n{e}")
            continue
    return True


def main():
    parser = argparse.ArgumentParser(
        prog="boe_scraper",
        description="Scrapes the board of elections NYC website for election data.",
    )
    parser.add_argument(
        "--url", type=str, help="The root level URL to query", default="https://enr.boenyc.gov"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        help="How often to poll the website and attempt to refresh data (seconds).",
        default=300,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="What folder to output the final jsons. If it's a s3 bucket, prefix the bucket name with 's3:'",
        default="../public",
    )
    parser.add_argument(
        "--dummy-data",
        action='store_true',
        help="Whether to use dummy data instead of actually pulling from the website. Useful for testing.",
    )

    args = parser.parse_args()
    assert (
        args.poll_interval >= 60
    ), "We shouldn't pull from the BOE website more than once a minute."
    while True:
        fetch_data(args)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
