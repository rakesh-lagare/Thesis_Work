import requests
import json
import csv
import argparse
import pandas as pd
import timeit

FPL_URL = "https://fantasy.premierleague.com/api/"
USER_SUMMARY_SUBURL = "element-summary/"
LEAGUE_CLASSIC_STANDING_SUBURL = "leagues-classic-standings/"
LEAGUE_H2H_STANDING_SUBURL = "leagues-h2h-standings/"
TEAM_ENTRY_SUBURL = "entry/"
PLAYERS_INFO_SUBURL = "bootstrap-static"
PLAYERS_INFO_FILENAME = "allPlayersInfo.json"

USER_SUMMARY_URL = FPL_URL + USER_SUMMARY_SUBURL
PLAYERS_INFO_URL = FPL_URL + PLAYERS_INFO_SUBURL
START_PAGE = 1
LEAGUE_ID = 271671



def getPlayersInfo():
    r = requests.get(PLAYERS_INFO_URL)
    jsonResponse = r.json()
    with open(PLAYERS_INFO_FILENAME, 'w') as outfile:
        json.dump(jsonResponse, outfile)

def getAllPlayersDetailedJson():
    with open(PLAYERS_INFO_FILENAME) as json_data:
        d = json.load(json_data)
        return d
