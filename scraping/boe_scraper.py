import time
import logging
from utils import get_elections, get_per_ed_results
import json
import sys

PRODUCTION_MODE = False
URL_TO_QUERY_PROD = "https://enr.boenyc.gov"
URL_TO_QUERY_TEST = "https://web.archive.org/web/20210623061537/https://web.enrboenyc.us/index.html"
POLL_INTERVAL_SECONDS = 60
URL_TO_QUERY = URL_TO_QUERY_PROD if PRODUCTION_MODE else URL_TO_QUERY_TEST
CONTEST_FILTER = ["mayor", "city council"]  # Makes scraper only look at races containing these terms. Leave empty list if you don't want to filter.
PARTY_FILTER = ["democratic"]  # Makes scraper only look at listed party primaries. Leave empty list if you don't want to filter.
CITY_COUNCIL_DISTRICTS = [38]


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def fetch_data():
    LOGGER.info("Fetching data.")
    try:
        elections_dict = get_elections(URL_TO_QUERY, LOGGER, CONTEST_FILTER, PARTY_FILTER)
        LOGGER.info(f"Success. elections_dict:\n{json.dumps(elections_dict, indent=4)}")
    except Exception as e:
        LOGGER.error(f"Error fetching data on get_elections({URL_TO_QUERY}):\n{e}")
        return False
    for election, link in elections_dict.items():
        try:
            results_dict = get_per_ed_results(link, LOGGER)
            fname = f"data/cache/{election}.json"
            with open(fname, "w") as f:
                json.dump(results_dict, f, indent=4)
            LOGGER.info(f"Success. Stored data in: {fname}")
        except Exception as e: 
            LOGGER.error(f"Error fetching data on get_per_ed_results({link}):\n{e}")
            continue
    return True


def main():
    assert POLL_INTERVAL_SECONDS >= 60, "We shouldn't pull from the BOE website more than once a minute."
    while True:
        fetch_data()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
