import time
import logging
from utils import *
import json
import sys
import argparse
import boto3
import tempfile
import os

# dsa races we care about.
BOE_ELECTION_WHITELIST = [
    {"position": "Mayor"}, # Zohran Mamdani
    {
        "position": "Member of the City Council",
        "sub_position": "Member of the City Council 38th Council District",
    }, # Alexa Avilés
]

# candidate name formatting.
TRANSLATION_DICT = {}
TRANSLATION_DICT["Member of the City Council 38th Council District"] = {
    "Alexa Aviles": "Alexa Avilés",
    "Luis E. Quero": "Luis E. Quero",
    "WRITE-IN": "WRITE-IN",
}
TRANSLATION_DICT["Mayor"] = {
    "Eric L. Adams (Independent)": "Eric L. Adams", 
    "Curtis A. Sliwa (Republican)": "Curtis A. Sliwa", 
    "Andrew M. Cuomo (Independent)": "Andrew M. Cuomo", 
    "Irene Estrada (Conservative)": "Irene Estrada",
    "Zohran Kwame Mamdani (Democratic)": "Zohran Kwame Mamdani", 
    "Joseph Hernandez (Independent)": "Joseph Hernandez", 
    "Jim Walden (Independent)": "Jim Walden", 
    "WRITE-IN": "WRITE-IN"
}

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def gen_dummy_data():
    elect_dists = json.load(open("data/shapes/districts.json", "r")).keys()
    return pd.DataFrame(
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
        if args.scrape_all_elections:
            whitelist = None
        else: 
            whitelist = BOE_ELECTION_WHITELIST
        elections_dict = get_elections(args.url, whitelist=whitelist)
        logger.info(f"Success. elections_dict:\n{json.dumps(elections_dict, indent=4)}")
    except Exception as e:
        logger.error(f"Error fetching data on get_elections({args.url}):\n{e}")
        return False
    for election, link in elections_dict.items():
        try:
            if args.dummy_data:
                results_df = gen_dummy_data()
                election += " (FAKE DATA)"
            else:
                results_df = get_election_results(election, link, format="df", candidate_rename_dict=TRANSLATION_DICT)
            if args.local_csv:
                results_df.to_csv(f"csv_data/{election}.csv")
            results_dict = get_grouped_dict(results_df)
            results_dict["last_updated"] = str(pd.Timestamp.now())

            fname = write_output(results_dict, election)
            logger.info(f"Sucess. Stored data in: {fname}")
        except Exception as e:
            logger.error(f"Error fetching data on get_per_ed_results({link}):\n{e}")
            continue
    return True


def main():
    parser = argparse.ArgumentParser(
        prog="boe_scraper",
        description="Scrapes the board of elections NYC website for election data.",
    )
    parser.add_argument(
        "--url", type=str, help="The root level URL to query", default="https://enr.boenyc.gov/index.html"
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
        default="public",
    )
    parser.add_argument(
        "--dummy-data",
        action='store_true',
        help="Whether to use dummy data instead of actually pulling from the website. Useful for testing.",
    )
    parser.add_argument(
        "--local-csv",
        action='store_true',
        help="Whether to also store a csv locally. (Useful if you want to upload things to drive.)"
    )

    parser.add_argument(
        "--scrape-all-elections", 
        action='store_true',
        help="Whether to ignore the whitelist and collect data on all elections",
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
