import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from io import StringIO
import numpy as np
import random
import time
import logging
import traceback


def safe_get_request(url, avg_wait_secs=1.0, timeout=None):
    """Requests from a url but ensures a random amount of time between requests.

    - Parameters
        - url: the url being requested.
        - avg_wait_secs: We will wait between 0.5-1.5x avg_wait_secs before requesting.
    - Returns
        - response: the output from requests.get(url)
    """
    time.sleep(avg_wait_secs * (0.5 + random.random()))
    return requests.get(url, timeout=timeout)


def append_dict(dict1, dict2):
    """Helper function to append dicts and assert no overlap."""
    assert (
        len(set(dict1.keys()) & set(dict2.keys())) == 0
    ), f"Found overlapping election names: {set(dict1.keys()) & set(dict2.keys())}"
    return dict1 | dict2


def matches_whitelist(whitelist, position=None, party=None, sub_position=None):
    """Helper function for whitelist."""
    if whitelist is None:
        # if no whitelist assume we want everything.
        return True
    matched_filter = False
    # if any filter in our whitelist matches proceed, else continue.
    for filter in whitelist:
        assert set(filter.keys()).issubset({"position", "party", "sub_position"})
        if "position" in filter and position and position != filter["position"]:
            continue
        if "party" in filter and party and party != filter["party"]:
            continue
        if "sub_position" in filter and sub_position and sub_position != filter["sub_position"]:
            continue
        matched_filter = True
    
    return matched_filter


def try_find_total_sublinks(sub_position, party, position, sub_link, url, whitelist):
    """Given a sublink you get from clicking 'AD Details*' in the main page, find the per-AD total page.

    - Parameters:
        - sub_position: ie Mayor, Member of the City Council etc.
        - party: empty string, Republican or Democratic
        - sub_link: The sub_url to scrape from.
        - url: The top level url to urljoin with. (Maybe this is redundant?)
        - whitelist: See the whitelist of get_elections
    - Returns:
        - output: a dictionary between election names, and the urls to pass to get_election_results.
    """
    # See if we're already on a page with a total elections link we can pass.
    cur_position = position if sub_position is None else sub_position
    sub_response = safe_get_request(sub_link)
    sub_soup = BeautifulSoup(sub_response.text, "html.parser")

    total_tag_links = [
        tag_link for tag_link in sub_soup.find_all("a", href=True) if tag_link.text == "Total"
    ]
    assert len(total_tag_links) < 2, f"Url: {sub_link} has more than one link with text 'Total'"
    if len(total_tag_links) == 1:
        election_name = cur_position if len(party) == 0 else f"{cur_position} ({party})"
        return {election_name: urljoin(url, total_tag_links[0]["href"])}

    # unfortunately we are not on the total elections page link, we have to recur.
    output = {}
    for subposition_link in sub_soup.find_all("a", href=True):
        subposition = subposition_link.text.strip()
        if subposition.startswith(position) and matches_whitelist(whitelist, sub_position=subposition, position=position, party=party):
            output = append_dict(
                output,
                try_find_total_sublinks(
                    sub_position=subposition,
                    party=party,
                    position=position,
                    sub_link=urljoin(url, subposition_link["href"]),
                    url=url,
                    whitelist=whitelist,
                ),
            )
    return output


def get_elections(url: str, logger: logging.Logger, whitelist=None):
    """Given the root url, ie https://enr.boenyc.gov, output a dict, from election name to per_ad link, ie:
    {"State Senator": "https://enr.boenyc.gov/CD27280AD0.html", ...}

    The values returned are able to be passed directly in get_per_ed_results below.

    TODO @chrispan-68: some sub-pages don't follow the format, ie: https://web.archive.org/web/20210625211942/https://web.enrboenyc.us/OF18AD0PY1.html

    - Parameters
        - url: root url of the boenyc election night results website.
        - whitelist: a list of dicts, specifying what races we're interested in, None means return all races, empty list is return no races. examples:
            {'party': 'Democratic'} <- only filters for democratic races (ie primaries)
            {'position': 'Member of the City Council', 'sub_position': 'Member of the City Council 7th Council District'} <- filters for 7th council district.
    - Returns
        - election_to_link: the dictionary described above.
    """
    response = safe_get_request(url)
    logger.info(f"Request completed: Code {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    soup_excerpt = soup.prettify()[:100]
    logger.info(f"Soup pulled. Excerpt:\n{soup_excerpt}")
    races_table = soup.find_all("table")[-1]  # this table contains each race with AD Details links.
    positions = []
    parties = []
    sub_links = []    
    logger.info("Generated races table.")
    rows = races_table.find_all('tr')
    logger.info("Split races table into rows.")
    row_counter = 0
    total_rows = len(rows)
    for row in rows:

        if row_counter > 0:
            logger.info(f"Processed {row_counter}/{total_rows} rows.")
        row_counter += 1
        cells = row.find_all('td')
        if len(cells) < 7:
            logger.info("Row doesn't match expected format, skipping")
            continue  # skip rows that don’t match expected format
        position = cells[2].get_text(strip=True)
        party = cells[3].get_text(strip=True)

        # check against our whitelist.
        if not matches_whitelist(whitelist, position=position, party=party):
            logger.info("Irrelevant race. Skipping.")
            continue

        ad_link_tag = cells[6].find("a")

        assert ad_link_tag, f"Couldn't find any links for row: {row.prettify()}"
        assert (
            ad_link_tag.text == "AD Details"
        ), f"Expected 'AD Details' as the text of the link, instead found: {ad_link_tag.text}"

        positions.append(position)
        parties.append(party)
        sub_links.append(urljoin(url, ad_link_tag["href"]))

    election_to_link = {}  # we still need to click on the 'Total' link in the AD Details page.
    for position, party, sub_link in zip(positions, parties, sub_links):
        outputs = try_find_total_sublinks(sub_position=None, party=party, position=position, sub_link=sub_link, url=url, whitelist=whitelist)
        election_to_link = append_dict(election_to_link, outputs)
    return election_to_link


def get_election_results(per_ad_url: str, logger: logging.Logger, format="grouped", incomplete_okay=False):
    """Given a url that points to the per AD totals for a race, return a dataframe with per ED results.
    This page should contain a table where each column is a candidate and the rows are Assembly Districts.

    This function looks for hrefs that start with the string 'AD'. If it can find no such links, then it throws an exception.

    - Parameters
        - per_ad_url: the url of the subpage with AD totals, ie: https://enr.boenyc.gov/CD27280AD0.html
        - format: 'dict' or 'df', depending on whether you want to do get_nested_dict post-formatting
        - incomplete_okay: sometimes, in the archive full results aren't cached, this toggles whether to return partial results or throw on any error.
    - Returns
        - df: Dataframe consisting of ElectDist (XXYY) where XX is AD, and YYY is the ED, with columns for each candidate.

    TODO @chrispan-68: Use regex for identifying the link.
    TODO @chrispan-68: Add asserts and readable errors.
    """
    response = safe_get_request(per_ad_url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    all_per_ad_links = soup.find_all("a", href=True)
    total_ad_links = len(all_per_ad_links)
    ad_counter = 0
    for link in all_per_ad_links:
        if ad_counter != 0:
            logger.info(f"{ad_counter}/{total_ad_links} links processed.")
        ad_counter += 1
        if link.text.startswith("AD"):
            subpage = urljoin(per_ad_url, link["href"])  # one AD's results.
            logger.info(f"Requesting {subpage}")
            subpage_r = safe_get_request(subpage)
            logger.info(f"Request complete. Request code = {subpage_r.status_code}")
            subpage_txt = subpage_r.text
            subpage_df = pd.read_html(StringIO(subpage_txt))[-1].dropna(
                axis=1, how="all"
            )
            columns = ["ED", "Reported %"] + list(
                subpage_df.iloc[0][2:] + " " + subpage_df.iloc[1][2:]
            )
            data.append(
                subpage_df.iloc[2:-1]
                .set_axis(columns, axis=1)
                .astype({col: int for col in columns[2:]})  # all vote counts are ints.
                .rename(columns=lambda col: col.replace("&nbsp", "").strip())
                .assign(
                    **{
                        "Reported %": lambda df: df["Reported %"].str[:-1].astype(float) / 100.0,
                        "AD": " ".join(link.text.split()),  # assembly district.
                    }
                )
                .assign(
                    AD=lambda df: df.AD.str.split().str[-1].astype(int),
                    ED=lambda df: df.ED.str.split().str[-1].astype(int),
                )
                .assign(
                    ElectDist=lambda df: df.AD * 1000
                    + df.ED  # AD * 1000 + ED (To match with geodata)
                )
            )
    df = pd.concat(data)
    if format == "df":
        return df
    elif format == "nested":
        output = get_nested_dict(df)
        output["last_updated"] = str(pd.Timestamp.now())
        return output
    elif format == "grouped":
        output = get_grouped_dict(df)
        output["last_updated"] = str(pd.Timestamp.now())
    else:
        raise ValueError("Unrecognized format", format)


def get_grouped_dict(df):
    """Useful helper function to get a dictionary output with several non-nested groups.
    - Parameters
        - column, the col to groupby on.
    - Returns
        - grouped_dict: a dictionary with several separate groups.
    """

    def _get_one_grouped_dict(df, column):
        assert (
            column in df.columns or column == "ALL"
        ), f"Column {column} not in dataframe columns {df.columns}."

        output = {}
        for g, gdf in df.assign(ALL="all").groupby(column):
            cand_df = gdf[
                [c for c in gdf.columns if not c in ["ED", "AD", "Reported %", "ElectDist", "ALL"]]
            ]
            total_voters = (
                cand_df.sum(axis=1) / gdf["Reported %"]
            )  # The total voters (including those who have not been reported)
            valid_eds = gdf.eval("`Reported %` > 0") & (
                cand_df.sum(axis=1) > 0
            )  # For EDs with 0% reporting, we should drop.

            group_res = {}
            group_res["total"] = float(cand_df.sum().sum())
            group_res["candidates"] = cand_df.sum().to_dict()
            if not valid_eds.any():
                group_res["approx_total_voters"] = 0
                group_res["reporting"] = 0
            else:
                group_res["approx_total_voters"] = float(total_voters.fillna(0).sum())
                group_res["reporting"] = float(
                    np.average(
                        gdf.loc[valid_eds]["Reported %"], weights=total_voters.loc[valid_eds]
                    )
                )
            output[g] = group_res

        return output

    DICT_GROUPS = ["ALL", "AD", "ElectDist"]
    GROUP_NAMES = ["all", "assembly_districts", "election_districts"]
    output = {}
    for group, name in zip(DICT_GROUPS, GROUP_NAMES):
        output[name] = _get_one_grouped_dict(df, group)
    return output


def get_nested_dict(
    df, subsets=[("AD", "assembly_districts"), ("ElectDist", "electoral_districts")]
):
    """Useful helper function to get a nested dictionary from a dataframe.
    - Parameters
        - subsets: defines all the nested subgroups and their names in the output dict.
    - Returns
        - nested_dict: a dictionary with len(subsets) levels of nesting and stats for each nesting group.
    """
    if len(subsets) == 0:
        cand_df = df[[c for c in df.columns if not c in ["ED", "AD", "Reported %", "ElectDist"]]]
        total_voters = (
            cand_df.sum(axis=1) / df["Reported %"]
        )  # The total voters (including those who have not been reported)
        valid_eds = df.eval("`Reported %` > 0") & (
            cand_df.sum(axis=1) > 0
        )  # For EDs with 0% reporting, we should drop.

        output = {}
        output["total"] = float(cand_df.sum().sum())
        output["candidates"] = cand_df.sum().to_dict()
        if not valid_eds.any():
            output["approx_total_voters"] = 0
            output["reporting"] = 0
        else:
            output["approx_total_voters"] = float(total_voters.fillna(0).sum())
            output["reporting"] = float(
                np.average(df.loc[valid_eds]["Reported %"], weights=total_voters.loc[valid_eds])
            )
        return output
    # recurse
    cur_subset = subsets[0]
    assert (
        len(cur_subset) == 2
    ), f"the subset: {cur_subset} is malformatted, every element of 'subsets' should be a pair."
    future_subsets = subsets[1:]
    output = {}
    output[cur_subset[1]] = {}
    output["total"] = 0
    output["approx_total_voters"] = 0
    output["candidates"] = {}
    sum_reporting_percs = 0.0
    for g, gdf in df.groupby(cur_subset[0]):
        child_dict = get_nested_dict(gdf, future_subsets)
        output[cur_subset[1]][g] = child_dict
        output["total"] += child_dict["total"]
        output["approx_total_voters"] += child_dict["approx_total_voters"]
        sum_reporting_percs += child_dict["approx_total_voters"] * child_dict["reporting"]
        for cand in child_dict["candidates"].keys():
            if not cand in output["candidates"]:
                output["candidates"][cand] = 0
            output["candidates"][cand] += child_dict["candidates"][cand]
    output["reporting"] = sum_reporting_percs / output["approx_total_voters"]
    return output
