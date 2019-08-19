import requests
import json
import csv
import argparse
import pandas as pd
import timeit


start = timeit.default_timer()

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

# Download all player data: https://fantasy.premiereague.com/drf/bootstrap-static
def getPlayersInfo():
    r = requests.get(PLAYERS_INFO_URL)
    jsonResponse = r.json()
    with open(PLAYERS_INFO_FILENAME, 'w') as outfile:
        json.dump(jsonResponse, outfile)

def getAllPlayersDetailedJson():
    with open(PLAYERS_INFO_FILENAME) as json_data:
        d = json.load(json_data)
        return d

        
getPlayersInfo()
playerElementIdToNameMap = {}
allPlayers = getAllPlayersDetailedJson()
teamsInfo=pd.DataFrame()
playerTypeInfo=pd.DataFrame()



for elementType in allPlayers["element_types"]:
    playerTypeInfo_temp = pd.DataFrame([[ elementType["id"] ,elementType["singular_name_short"],elementType["singular_name"]]],
                                       columns=['id','singular_name_short','singular_name']   ) 
    playerTypeInfo = playerTypeInfo.append(playerTypeInfo_temp,ignore_index=True)


for team in allPlayers["teams"]:
    teamsInfo_temp = pd.DataFrame([[ team["id"] ,team["name"],team["short_name"]]],  columns=['id','name','short_name']   ) 
    teamsInfo = teamsInfo.append(teamsInfo_temp,ignore_index=True)

 
playerInfo = pd.DataFrame()
for element in allPlayers["elements"]:
    teamName_temp = teamsInfo.loc[teamsInfo['id'] == element["team"]  ]
    teamName = teamName_temp.iloc[0]['short_name']
    #print(teamName)
    playerDF_temp = pd.DataFrame([[element["id"], element["web_name"],element["form"],element["now_cost"]/10,element["total_points"],element["points_per_game"],element["selected_by_percent"],element["goals_scored"],element["minutes"],   teamName ]], 
                                columns=['id','web_name','form',"now_cost","total_points","points_per_game","selected_by_percent","goals_scored",'minutes',  'team_name'])
    playerInfo = playerInfo.append(playerDF_temp,ignore_index=True)


playerInfo.to_csv("playerInfo.csv", index=False)

stop = timeit.default_timer()
print('Time: ', stop - start)  