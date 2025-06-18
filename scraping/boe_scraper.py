import time
import logging
from utils import get_elections, get_per_ed_results
import json
import sys

URL_TO_QUERY = "https://enr.boenyc.gov"
POLL_INTERVAL_SECONDS = 60


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def fetch_data():
    try:
        elections_dict = get_elections(URL_TO_QUERY)
        logger.info(f"Success. elections_dict:\n{json.dumps(elections_dict, indent=4)}")
    except Exception as e:
        logger.error(f"Error fetching data on get_elections({URL_TO_QUERY}):\n{e}")
        return False
    for election, link in elections_dict.items():
        try:
            results_dict = get_per_ed_results(link)
            fname = f"data/cache/{election}.json"
            with open(fname, "w") as f:
                json.dump(results_dict, f, indent=4)
            logger.info(f"Sucess. Stored data in: {fname}")
        except Exception as e: 
            logger.error(f"Error fetching data on get_per_ed_results({link}):\n{e}")
            continue
    return True


def main():
    assert POLL_INTERVAL_SECONDS >= 60, "We shouldn't pull from the BOE website more than once a minute."
    while True:
        fetch_data()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
