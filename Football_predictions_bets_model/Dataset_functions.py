#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

def clean_seasonal_data(dataset):
    dataset.drop(['HTHG','HTAG','HTR','Referee','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA','AvgH','AvgD',	'AvgA',	'B365>2.5',	'B365<2.5',	'P>2.5',	'P<2.5',	'Max>2.5',	'Max<2.5',	'Avg>2.5',	'Avg<2.5',	'AHh',	'B365AHH',	'B365AHA',	'PAHH',	'PAHA',	'MaxAHH',	'MaxAHA',	'AvgAHH',	'AvgAHA',	'B365CH',	'B365CD',	'B365CA',	'BWCH',	'BWCD',	'BWCA',	'IWCH',	'IWCD',	'IWCA',	'PSCH',	'PSCD',	'PSCA',	'WHCH',	'WHCD',	'WHCA',	'VCCH',	'VCCD',	'VCCA',	'MaxCH',	'MaxCD',	'MaxCA',	'AvgCH',	'AvgCD',	'AvgCA',	'B365C>2.5',	'B365C<2.5',	'PC>2.5',	'PC<2.5',	'MaxC>2.5',	'MaxC<2.5',	'AvgC>2.5',	'AvgC<2.5',	'AHCh',	'B365CAHH',	'B365CAHA',	'PCAHH',	'PCAHA',	'MaxCAHH',	'MaxCAHA',	'AvgCAHH',	'AvgCAHA','BFH',	'BFD',	'BFA',	'1XBH',	'1XBD',	'1XBA',	'BFEH',	'BFED',	'BFEA',	'BFE>2.5',	'BFE<2.5',	'BFEAHH',	'BFEAHA',	'BFCH',	'BFCD',	'BFCA'	'1XBCH',	'1XBCD',	'1XBCA',	'BFECH',	'BFECD',	'BFECA',	'BFEC>2.5',	'BFEC<2.5',	'BFECAHH',	'BFECAHA'	 ], inplace =True, axis = 1, errors='ignore')
    dataset = pd.get_dummies(dataset, columns=['HomeTeam', 'AwayTeam'])
    dummy_columns = dataset.select_dtypes(bool).columns
    dataset[dummy_columns] = dataset[dummy_columns].astype(int)
    dataset['Date'] = pd.to_datetime(dataset['Date'],format = '%d/%m/%Y' )
    return dataset

def clean_v2(dataset):
    dataset.drop(['HTHG','HTAG','HTR','Referee','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA','AvgH','AvgD',	'AvgA',	'B365>2.5',	'B365<2.5',	'P>2.5',	'P<2.5',	'Max>2.5',	'Max<2.5',	'Avg>2.5',	'Avg<2.5',	'AHh',	'B365AHH',	'B365AHA',	'PAHH',	'PAHA',	'MaxAHH',	'MaxAHA',	'AvgAHH',	'AvgAHA',	'B365CH',	'B365CD',	'B365CA',	'BWCH',	'BWCD',	'BWCA',	'IWCH',	'IWCD',	'IWCA',	'PSCH',	'PSCD',	'PSCA',	'WHCH',	'WHCD',	'WHCA',	'VCCH',	'VCCD',	'VCCA',	'MaxCH',	'MaxCD',	'MaxCA',	'AvgCH',	'AvgCD',	'AvgCA',	'B365C>2.5',	'B365C<2.5',	'PC>2.5',	'PC<2.5',	'MaxC>2.5',	'MaxC<2.5',	'AvgC>2.5',	'AvgC<2.5',	'AHCh',	'B365CAHH',	'B365CAHA',	'PCAHH',	'PCAHA',	'MaxCAHH',	'MaxCAHA',	'AvgCAHH',	'AvgCAHA','BFH',	'BFD',	'BFA',	'1XBH',	'1XBD',	'1XBA',	'BFEH',	'BFED',	'BFEA',	'BFE>2.5',	'BFE<2.5',	'BFEAHH',	'BFEAHA',	'BFCH',	'BFCD',	'BFCA'	'1XBCH',	'1XBCD',	'1XBCA',	'BFECH',	'BFECD',	'BFECA',	'BFEC>2.5',	'BFEC<2.5',	'BFECAHH',	'BFECAHA'	 ], inplace =True, axis = 1, errors='ignore')
    return dataset
    
def get_team_matches(team_name, dataset):
    team_games =  dataset[dataset[dataset.filter(like=team_name).columns].gt(0.5).any(axis=1)]
    matchday_array = np.arange(1, len(team_games) + 1)
    team_games['matchday'] = matchday_array
    return team_games

def get_team_home_matches(team_name, dataset):
    team_matches = get_team_matches(team_name, dataset)
    team_home_games =  team_matches[team_matches[team_matches.filter(like=f"HomeTeam_{team_name}").columns].gt(0.5).any(axis=1)]
    return team_home_games

def get_team_away_matches(team_name, dataset):
    team_matches = get_team_matches(team_name, dataset)
    team_away_games =  team_matches[team_matches[team_matches.filter(like=f"AwayTeam_{team_name}").columns].gt(0.5).any(axis=1)]
    return team_away_games

def last_n_matches_form(team_games, team_name, n):
    team_form = [0]
    last_points =[0]
    for index, row in team_games.iterrows():
        if (row['FTR'] == 'H' and row[f"HomeTeam_{team_name}"] == 1) or (row['FTR'] == 'A' and row[f"AwayTeam_{team_name}"] == 1):
            last_points.append(3)
        elif row['FTR'] == 'D':
            last_points.append(1)
        else:
            last_points.append(0)
        if len(last_points)>n:
            del last_points[0]
        team_form.append(np.sum(last_points))
    del team_form[-1]
    return team_form


def team_current_points_total(team_name, dataset, which = 'all'):
    team_points = [0]
    last_points =[0]
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if (row['FTR'] == 'H' and row[f"HomeTeam_{team_name}"] == 1) or (row['FTR'] == 'A' and row[f"AwayTeam_{team_name}"] == 1):
                last_points.append(3)
            elif row['FTR'] == 'D':
                last_points.append(1)
            else:
                last_points.append(0)
            team_points.append(np.sum(last_points))     
    elif which == 'home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row['FTR'] == 'H':
                last_points.append(3)
            elif row['FTR'] == 'D':
                last_points.append(1)
            else:
                last_points.append(0)
            team_points.append(np.sum(last_points))
    elif which == 'away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row['FTR'] == 'A':
                last_points.append(3)
            elif row['FTR'] == 'D':
                last_points.append(1)
            else:
                last_points.append(0)
            team_points.append(np.sum(last_points))
    del team_points[-1]
    return team_points


def get_team_form(team_name, dataset, n, which = 'all'):
    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    form_n_matches = last_n_matches_form(team_games, team_name, n)
    return form_n_matches


def get_team_goals_total(team_name, dataset, which = 'all'):
    team_goals = [0]
    goals_against = [0]
    last_goals =[0]
    last_goals_against = [0]
    team_goals_per_game = [0]
    against_goals_per_game = [0]
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row[f"HomeTeam_{team_name}"] == 1 :
                last_goals.append(row['FTHG'])
                last_goals_against.append(row['FTAG'])
            elif row[f"AwayTeam_{team_name}"] == 1:
                last_goals.append(row['FTAG'])
                last_goals_against.append(row['FTHG'])
            team_goals.append(np.sum(last_goals))
            goals_against.append(np.sum(last_goals_against))
            team_goals_per_game.append(team_goals[-1]/len(team_goals))
            against_goals_per_game.append(goals_against[-1]/len(goals_against))
    elif which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_goals.append(row['FTHG'])
            last_goals_against.append(row['FTAG'])
            team_goals.append(np.sum(last_goals))
            goals_against.append(np.sum(last_goals_against))
            team_goals_per_game.append(team_goals[-1]/len(team_goals))
            against_goals_per_game.append(goals_against[-1]/len(goals_against))
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_goals.append(row['FTAG'])
            last_goals_against.append(row['FTHG'])
            team_goals.append(np.sum(last_goals))
            goals_against.append(np.sum(last_goals_against))
            team_goals_per_game.append(team_goals[-1]/len(team_goals))
            against_goals_per_game.append(goals_against[-1]/len(goals_against))
    del team_goals[-1]
    del goals_against[-1]
    del team_goals_per_game[-1]
    del against_goals_per_game[-1]
    return team_goals, goals_against, team_goals_per_game, against_goals_per_game


def last_n_matches_goals(team_games, team_name, n):
    team_goals = [0]
    goals_against = [0]
    last_goals =[0]
    last_goals_against = [0]
    
    for index, row in team_games.iterrows():
        if row[f"HomeTeam_{team_name}"] == 1:
            last_goals.append(row['FTHG'])
            last_goals_against.append(row['FTAG'])
        elif row[f"AwayTeam_{team_name}"] == 1:
            last_goals.append(row['FTAG'])
            last_goals_against.append(row['FTHG'])
        if len(last_goals)>n:
            del last_goals[0]
            del last_goals_against[0]
        team_goals.append(np.sum(last_goals))
        goals_against.append(np.sum(last_goals_against))
    del team_goals[-1]
    del goals_against[-1]
    return team_goals, goals_against

def get_team_goals_n_matches(team_name, dataset, n, which = 'all'):
    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    team_goals_n_matches, goals_against_n_matches = last_n_matches_goals(team_games, team_name, n)
    return  team_goals_n_matches, goals_against_n_matches


def get_team_shots_per_game(team_name, dataset, which = 'all', on_target = False):
    team_shots = [0]
    shots_against = [0]
    last_shots =[0]
    last_shots_against = [0]
    team_shots_per_game = [0]
    against_shots_per_game = [0]
    if on_target == False:
        home_shots = 'HS'
        away_shots = 'AS'
    else:
        home_shots = 'HST'
        away_shots = 'AST'
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row[f"HomeTeam_{team_name}"] == 1 :
                last_shots.append(row[home_shots])
                last_shots_against.append(row[away_shots])
            elif row[f"AwayTeam_{team_name}"] == 1:
                last_shots.append(row[away_shots])
                last_shots_against.append(row[home_shots])
            team_shots.append(np.sum(last_shots))
            shots_against.append(np.sum(last_shots_against))
            team_shots_per_game.append(team_shots[-1]/len(team_shots))
            against_shots_per_game.append(shots_against[-1]/len(shots_against))
    elif which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_shots.append(row[home_shots])
            last_shots_against.append(row[away_shots])
            team_shots.append(np.sum(last_shots))
            shots_against.append(np.sum(last_shots_against))
            team_shots_per_game.append(team_shots[-1]/len(team_shots))
            against_shots_per_game.append(shots_against[-1]/len(shots_against))
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_shots.append(row[away_shots])
            last_shots_against.append(row[home_shots])
            team_shots.append(np.sum(last_shots))
            shots_against.append(np.sum(last_shots_against))
            team_shots_per_game.append(team_shots[-1]/len(team_shots))
            against_shots_per_game.append(shots_against[-1]/len(shots_against))
    del team_shots[-1]
    del shots_against[-1]
    del team_shots_per_game[-1]
    del against_shots_per_game[-1]
    return team_shots_per_game, against_shots_per_game


def last_n_matches_shots(team_games, team_name, n, on_target = False):
    team_shots = [0]
    shots_against = [0]
    last_shots =[0]
    last_shots_against = [0]
    if on_target == False:
        home_shots = 'HS'
        away_shots = 'AS'
    else:
        home_shots = 'HST'
        away_shots = 'AST'
    for index, row in team_games.iterrows():
        if row[f"HomeTeam_{team_name}"] == 1:
            last_shots.append(row[home_shots])
            last_shots_against.append(row[away_shots])
        elif row[f"AwayTeam_{team_name}"] == 1:
            last_shots.append(row[away_shots])
            last_shots_against.append(row[home_shots])
        if len(last_shots)>n:
            del last_shots[0]
            del last_shots_against[0]
        team_shots.append(np.sum(last_shots))
        shots_against.append(np.sum(last_shots_against))
    del team_shots[-1]
    del shots_against[-1]
    return team_shots, shots_against

def get_team_shots_n_matches(team_name, dataset, n, which = 'all', on_target = False):
    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    team_shots_n_matches, shots_against_n_matches = last_n_matches_shots(team_games, team_name, n, on_target = on_target)
    return  team_shots_n_matches, shots_against_n_matches


def get_team_cards_per_game(team_name, dataset, which = 'all', yellow = True):
    team_cards = [0]
    cards_against = [0]
    last_cards =[0]
    last_cards_against = [0]
    team_cards_per_game = [0]
    against_cards_per_game = [0]
    if yellow == True:
        home_cards = 'HY'
        away_cards = 'AY'
    else:
        home_cards = 'HR'
        away_cards = 'AR'
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row[f"HomeTeam_{team_name}"] == 1 :
                last_cards.append(row[home_cards])
                last_cards_against.append(row[away_cards])
            elif row[f"AwayTeam_{team_name}"] == 1:
                last_cards.append(row[away_cards])
                last_cards_against.append(row[home_cards])
            team_cards.append(np.sum(last_cards))
            cards_against.append(np.sum(last_cards_against))
            team_cards_per_game.append(team_cards[-1]/len(team_cards))
            against_cards_per_game.append(cards_against[-1]/len(cards_against))
    elif which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_cards.append(row[home_cards])
            last_cards_against.append(row[away_cards])
            team_cards.append(np.sum(last_cards))
            cards_against.append(np.sum(last_cards_against))
            team_cards_per_game.append(team_cards[-1]/len(team_cards))
            against_cards_per_game.append(cards_against[-1]/len(cards_against))
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_cards.append(row[away_cards])
            last_cards_against.append(row[home_cards])
            team_cards.append(np.sum(last_cards))
            cards_against.append(np.sum(last_cards_against))
            team_cards_per_game.append(team_cards[-1]/len(team_cards))
            against_cards_per_game.append(cards_against[-1]/len(cards_against))
    del team_cards[-1]
    del cards_against[-1]
    del team_cards_per_game[-1]
    del against_cards_per_game[-1]
    return team_cards_per_game, against_cards_per_game

def get_team_xg_per_game(team_name, dataset, which = 'all'):
    team_shots = [0]
    shots_against = [0]
    last_shots =[0]
    last_shots_against = [0]
    team_shots_per_game = [0]
    against_shots_per_game = [0]
    home_shots = 'Home xG'
    away_shots = 'Away xG'
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row[f"HomeTeam_{team_name}"] == 1 :
                last_shots.append(row[home_shots])
                last_shots_against.append(row[away_shots])
            elif row[f"AwayTeam_{team_name}"] == 1:
                last_shots.append(row[away_shots])
                last_shots_against.append(row[home_shots])
            team_shots.append(np.sum(last_shots))
            shots_against.append(np.sum(last_shots_against))
            team_shots_per_game.append(team_shots[-1]/len(team_shots))
            against_shots_per_game.append(shots_against[-1]/len(shots_against))
    elif which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_shots.append(row[home_shots])
            last_shots_against.append(row[away_shots])
            team_shots.append(np.sum(last_shots))
            shots_against.append(np.sum(last_shots_against))
            team_shots_per_game.append(team_shots[-1]/len(team_shots))
            against_shots_per_game.append(shots_against[-1]/len(shots_against))
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_shots.append(row[away_shots])
            last_shots_against.append(row[home_shots])
            team_shots.append(np.sum(last_shots))
            shots_against.append(np.sum(last_shots_against))
            team_shots_per_game.append(team_shots[-1]/len(team_shots))
            against_shots_per_game.append(shots_against[-1]/len(shots_against))
    del team_shots[-1]
    del shots_against[-1]
    del team_shots_per_game[-1]
    del against_shots_per_game[-1]
    return team_shots_per_game, against_shots_per_game


def last_n_matches_xg(team_games, team_name, n):
    team_shots = [0]
    shots_against = [0]
    last_shots =[0]
    last_shots_against = [0]
    home_shots = 'Home xG'
    away_shots = 'Away xG'
    for index, row in team_games.iterrows():
        if row[f"HomeTeam_{team_name}"] == 1:
            last_shots.append(row[home_shots])
            last_shots_against.append(row[away_shots])
        elif row[f"AwayTeam_{team_name}"] == 1:
            last_shots.append(row[away_shots])
            last_shots_against.append(row[home_shots])
        if len(last_shots)>n:
            del last_shots[0]
            del last_shots_against[0]
        team_shots.append(np.sum(last_shots))
        shots_against.append(np.sum(last_shots_against))
    del team_shots[-1]
    del shots_against[-1]
    return team_shots, shots_against

def get_team_xg_n_matches(team_name, dataset, n, which = 'all'):
    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    team_shots_n_matches, shots_against_n_matches = last_n_matches_shots(team_games, team_name, n)
    return  team_shots_n_matches, shots_against_n_matches



import requests
from bs4 import BeautifulSoup

# URL of the website to scrape


def scrape_data(url, season = '2023-2024'):
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # Find the table with match data
        table = soup.find('table', {'id': f'sched_{season}_9_1'})
    
        # Initialize a list to hold match data
        matches = []
    
        # Iterate through the rows of the table
        for row in table.find_all('tr'):
            # Get all the columns in the row
            cols = row.find_all('td')
    
            # Check if the row contains match data
            if len(cols) > 0:
                # Extract relevant data from the columns
                date = cols[1].text.strip()
                home_team = cols[3].text.strip()
                home_xg = cols[4].text.strip()
                away_xg = cols[6].text.strip()
                away_team = cols[7].text.strip()
                if len(date)>5:
                    matches.append((date, home_team, away_team, home_xg, away_xg))
                
                
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
    xgoals = pd.DataFrame(matches, columns=['Date', 'Home Team', 'Away Team', 'Home xG', 'Away xG'])
    return xgoals

def join_scraped(url, season_data, season, teams):
    xgoals= scrape_data(url, season)
    xgoals = xgoals[['Home Team', 'Away Team','Home xG', 'Away xG']]

    teams = sorted(teams)
    teams2 = sorted(xgoals['Home Team'].unique())

    dict_teams = {}
    for i in range(len(teams2)):
        dict_teams[teams2[i]] = teams[i]
    xgoals = xgoals.replace(dict_teams) 
    season_data = pd.merge(season_data, xgoals, left_on = ['HomeTeam', 'AwayTeam'], right_on = ['Home Team', 'Away Team'])
    season_data['Home xG'] = pd.to_numeric(season_data['Home xG'], errors='coerce')
    season_data['Away xG'] = pd.to_numeric(season_data['Away xG'], errors='coerce')
    xgoals = season_data[['Home xG','Away xG']]
    season_data = clean_seasonal_data(season_data)
    season_data[['Home xG','Away xG']] = xgoals
    return season_data

def create_seasonal_table(season_data, teams):
    teams_datasets_seasonal = []
    for i in range(len(teams)):
        team_name = teams[i]
        team_1_games =  get_team_matches(team_name, season_data)
        team_1_away_games = get_team_away_matches(team_name, season_data)
        team_1_home_games = get_team_home_matches(team_name, season_data)
        
        team_1_games['total_points'] = team_current_points_total(team_name, season_data)
        team_1_games['total_points_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_points'], np.nan)
        team_1_games['total_points_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_points'], np.nan)
        
        team_1_games['5_form'] = get_team_form(team_name, season_data, 5, 'all')
        team_1_games['5_form_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form'], np.nan)
        team_1_games['5_form_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form'], np.nan)
        
        team_1_games['10_form'] = get_team_form(team_name, season_data, 10, 'all')
        team_1_games['10_form_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form'], np.nan)
        team_1_games['10_form_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form'], np.nan)
        
        team_1_all_goals_stats = get_team_goals_total(team_name, season_data, 'all')
        team_1_home_goals_stats = get_team_goals_total(team_name, season_data, 'home')
        team_1_away_goals_stats = get_team_goals_total(team_name, season_data, 'away')
        
        team_1_games['total_goals'] = team_1_all_goals_stats[0]
        team_1_games['total_goals_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_goals'], np.nan)
        team_1_games['total_goals_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_goals'], np.nan)
        
        team_1_games['total_goals_against'] = team_1_all_goals_stats[1]
        team_1_games['total_goals_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_goals_against'], np.nan)
        team_1_games['total_goals_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_goals_against'], np.nan)
        
        team_1_games['total_goals_per_game'] = team_1_all_goals_stats[2]
        team_1_games['total_goals_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_goals_per_game'], np.nan)
        team_1_games['total_goals_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_goals_per_game'], np.nan)
        
        team_1_games['total_goals_against_per_game'] = team_1_all_goals_stats[3]
        team_1_games['total_goals_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_goals_against_per_game'], np.nan)
        team_1_games['total_goals_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_goals_against_per_game'], np.nan)
        
        team_1_games['5_form_goals_scored'] = get_team_goals_n_matches(team_name, season_data, 5, 'all')[0]
        team_1_games['5_form_goals_scored_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_goals_scored'], np.nan)
        team_1_games['5_form_goals_scored_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_goals_scored'], np.nan)
        
        
        team_1_games['5_form_goals_against'] = get_team_goals_n_matches(team_name, season_data, 5, 'all')[1]
        team_1_games['5_form_goals_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_goals_against'], np.nan)
        team_1_games['5_form_goals_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_goals_against'], np.nan)
        
        team_1_games['10_form_goals_scored'] = get_team_goals_n_matches(team_name, season_data, 10, 'all')[0]
        team_1_games['10_form_goals_scored_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_goals_scored'], np.nan)
        team_1_games['10_form_goals_scored_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_goals_scored'], np.nan)
        
        team_1_games['10_form_goals_against'] = get_team_goals_n_matches(team_name, season_data, 10, 'all')[1]
        team_1_games['10_form_goals_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_goals_against'], np.nan)
        team_1_games['10_form_goals_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_goals_against'], np.nan)
        
        
        team_1_games['total_shots_per_game'] = get_team_shots_per_game(team_name, season_data, 'all')[0]
        team_1_games['total_shots_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_shots_per_game'], np.nan)
        team_1_games['total_shots_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_shots_per_game'], np.nan)
        
        team_1_games['total_shots_against_per_game'] = get_team_shots_per_game(team_name, season_data, 'all')[1]
        team_1_games['total_shots_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_shots_against_per_game'], np.nan)
        team_1_games['total_shots_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_shots_against_per_game'], np.nan)
        
        team_1_games['total_shots_on_target_per_game'] = get_team_shots_per_game(team_name, season_data, 'all', on_target = True)[0]
        team_1_games['total_shots_on_target_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_shots_on_target_per_game'], np.nan)
        team_1_games['total_shots_on_target_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_shots_on_target_per_game'], np.nan)
        
        team_1_games['total_shots_on_target_against_per_game'] = get_team_shots_per_game(team_name, season_data, 'all', on_target = True)[1]
        team_1_games['total_shots_on_target_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_shots_on_target_against_per_game'], np.nan)
        team_1_games['total_shots_on_target_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_shots_on_target_against_per_game'], np.nan)
        
        
        team_1_games['5_form_shots'] = get_team_shots_n_matches(team_name, season_data, 5, 'all', on_target = False)[0]
        team_1_games['5_form_shots_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_shots'], np.nan)
        team_1_games['5_form_shots_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_shots'], np.nan)
        
        
        team_1_games['5_form_shots_against'] = get_team_shots_n_matches(team_name, season_data, 5, 'all', on_target = False)[1]
        team_1_games['5_form_shots_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_shots_against'], np.nan)
        team_1_games['5_form_shots_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_shots_against'], np.nan)
        
        team_1_games['10_form_shots'] = get_team_shots_n_matches(team_name, season_data, 10, 'all', on_target = False)[0]
        team_1_games['10_form_shots_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_shots_against'], np.nan)
        team_1_games['10_form_shots_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_shots'], np.nan)
        
        team_1_games['10_form_shots_against'] = get_team_shots_n_matches(team_name, season_data, 10, 'all', on_target = False)[1]
        team_1_games['10_form_shots_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_shots_against'], np.nan)
        team_1_games['10_form_shots_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_shots_against'], np.nan)
        
        team_1_games['5_form_shots_on_target'] = get_team_shots_n_matches(team_name, season_data, 5, 'all', on_target = True)[0]
        team_1_games['5_form_shots_on_target_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_shots_on_target'], np.nan)
        team_1_games['5_form_shots_on_target_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_shots_on_target'], np.nan)
        
        team_1_games['5_form_shots_on_target_against'] = get_team_shots_n_matches(team_name, season_data, 5, 'all', on_target = True)[1]
        team_1_games['5_form_shots_on_target_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_shots_on_target_against'], np.nan)
        team_1_games['5_form_shots_on_target_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_shots_on_target_against'], np.nan)
        
        team_1_games['10_form_shots_on_target'] = get_team_shots_n_matches(team_name, season_data, 10, 'all', on_target = True)[0]
        team_1_games['10_form_shots_on_target_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_shots_on_target'], np.nan)
        team_1_games['10_form_shots_on_target_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_shots_on_target'], np.nan)
        
        
        team_1_games['10_form_shots_on_target_against'] = get_team_shots_n_matches(team_name, season_data, 10, 'all', on_target = True)[1]
        team_1_games['10_form_shots_on_target_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_shots_on_target_against'], np.nan)
        team_1_games['10_form_shots_on_target_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_shots_on_target_against'], np.nan)
        
        
        team_1_games['total_yellow_cards_per_game'] = get_team_cards_per_game(team_name, season_data, 'all', yellow = True)[0]
        team_1_games['total_yellow_cards_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_yellow_cards_per_game'], np.nan)
        team_1_games['total_yellow_cards_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_yellow_cards_per_game'], np.nan)
        
        
        team_1_games['total_yellow_cards_against_per_game'] = get_team_cards_per_game(team_name, season_data, 'all', yellow = True)[1]
        team_1_games['total_yellow_cards_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_yellow_cards_against_per_game'], np.nan)
        team_1_games['total_yellow_cards_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_yellow_cards_against_per_game'], np.nan)
        
        
        team_1_games['total_red_cards_per_game'] = get_team_cards_per_game(team_name, season_data, 'all', yellow = False)[0]
        team_1_games['total_red_cards_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_red_cards_per_game'], np.nan)
        team_1_games['total_red_cards_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_red_cards_per_game'], np.nan)
        
        team_1_games['total_red_cards_against_per_game'] = get_team_cards_per_game(team_name, season_data, 'all', yellow = False)[1]
        team_1_games['total_red_cards_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_red_cards_against_per_game'], np.nan)
        team_1_games['total_red_cards_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_red_cards_against_per_game'], np.nan)
        
        
        team_1_games['total_xg_per_game'] = get_team_xg_per_game(team_name, season_data, 'all')[0]
        team_1_games['total_xg_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_xg_per_game'], np.nan)
        team_1_games['total_xg_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_xg_per_game'], np.nan)
        
        team_1_games['total_xg_against_per_game'] = get_team_xg_per_game(team_name, season_data, 'all')[1]
        team_1_games['total_xg_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_xg_against_per_game'], np.nan)
        team_1_games['total_xg_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_xg_against_per_game'], np.nan)
        
        team_1_games['5_form_xg'] = get_team_xg_n_matches(team_name, season_data, 5, 'all')[0]
        team_1_games['5_form_xg_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_xg'], np.nan)
        team_1_games['5_form_xg_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_xg'], np.nan)
        
        team_1_games['5_form_xg_against'] = get_team_xg_n_matches(team_name, season_data, 5, 'all')[1]
        team_1_games['5_form_xg_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['5_form_xg_against'], np.nan)
        team_1_games['5_form_xg_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['5_form_xg_against'], np.nan)
        
        
        team_1_games['10_form_xg'] = get_team_xg_n_matches(team_name, season_data, 10, 'all')[0]
        team_1_games['10_form_xg_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_xg'], np.nan)
        team_1_games['10_form_xg_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_xg'], np.nan)
        
        team_1_games['10_form_xg_against'] = get_team_xg_n_matches(team_name, season_data, 10, 'all')[1]
        team_1_games['10_form_xg_against_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['10_form_xg_against'], np.nan)
        team_1_games['10_form_xg_against_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['10_form_xg_against'], np.nan)
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_points_away'] = team_current_points_total(team_name, season_data, 'away')
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_points_home'] = team_current_points_total(team_name, season_data, 'home')
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_away'] = get_team_form(team_name, season_data, 5, 'away')
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_away'] = get_team_form(team_name, season_data, 10, 'away')
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_home'] = get_team_form(team_name, season_data, 5, 'home')
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_home'] = get_team_form(team_name, season_data, 10, 'home')
        
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_goals_away'] = team_1_away_goals_stats[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_goals_against_away'] = team_1_away_goals_stats[1]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_goals_per_game_away'] = team_1_away_goals_stats[2]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_goals_against_per_game_away'] = team_1_away_goals_stats[3]
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_goals_home'] = team_1_home_goals_stats[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_goals_against_home'] = team_1_home_goals_stats[1]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_goals_per_game_home'] = team_1_home_goals_stats[2]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_goals_against_per_game_home'] = team_1_home_goals_stats[3]
        
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_goals_scored_away'] = get_team_goals_n_matches(team_name, season_data, 5, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_goals_scored_away'] = get_team_goals_n_matches(team_name, season_data, 10, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_goals_against_away'] = get_team_goals_n_matches(team_name, season_data, 5, 'away')[1]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_goals_against_away'] = get_team_goals_n_matches(team_name, season_data, 10, 'away')[1]
        
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_goals_scored_home'] = get_team_goals_n_matches(team_name, season_data, 5, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_goals_scored_home'] = get_team_goals_n_matches(team_name, season_data, 10, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_goals_against_home'] = get_team_goals_n_matches(team_name, season_data, 5, 'home')[1]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_goals_against_home'] = get_team_goals_n_matches(team_name, season_data, 10, 'home')[1]
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_shots_per_game_home'] = get_team_shots_per_game(team_name, season_data, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_shots_against_per_game_home'] = get_team_shots_per_game(team_name, season_data, 'home')[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_shots_per_game_away'] = get_team_shots_per_game(team_name, season_data, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_shots_against_per_game_away'] = get_team_shots_per_game(team_name, season_data, 'away')[1]
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_shots_on_target_per_game_home'] = get_team_shots_per_game(team_name, season_data, 'home', on_target = True)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_shots_on_target_against_per_game_home'] = get_team_shots_per_game(team_name, season_data, 'home', on_target = True)[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_shots_on_target_per_game_away'] = get_team_shots_per_game(team_name, season_data, 'away', on_target = True)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_shots_on_target_against_per_game_away'] = get_team_shots_per_game(team_name, season_data, 'away', on_target = True)[1]
        
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_shots_away'] = get_team_shots_n_matches(team_name, season_data, 5, 'away', on_target = False)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_shots_away'] = get_team_shots_n_matches(team_name, season_data, 10, 'away', on_target = False)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_shots_against_away'] = get_team_shots_n_matches(team_name, season_data, 5, 'away', on_target = False)[1]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_shots_against_away'] = get_team_shots_n_matches(team_name, season_data, 10, 'away', on_target = False)[1]
        
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_shots_home'] = get_team_shots_n_matches(team_name, season_data, 5, 'home', on_target = False)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_shots_home'] = get_team_shots_n_matches(team_name, season_data, 10, 'home', on_target = False)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_shots_against_home'] = get_team_shots_n_matches(team_name, season_data, 5, 'home', on_target = False)[1]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_shots_against_home'] = get_team_shots_n_matches(team_name, season_data, 10, 'home', on_target = False)[1]
        
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_shots_on_target_away'] = get_team_shots_n_matches(team_name, season_data, 5, 'away', on_target = True)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_shots_on_target_away'] = get_team_shots_n_matches(team_name, season_data, 10, 'away', on_target = True)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_shots_on_target_against_away'] = get_team_shots_n_matches(team_name, season_data, 5, 'away', on_target = True)[1]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_shots_on_target_against_away'] = get_team_shots_n_matches(team_name, season_data, 10, 'away', on_target = True)[1]
        
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_shots_on_target_home'] = get_team_shots_n_matches(team_name, season_data, 5, 'home', on_target = True)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_shots_on_target_home'] = get_team_shots_n_matches(team_name, season_data, 10, 'home', on_target = True)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_shots_on_target_against_home'] = get_team_shots_n_matches(team_name, season_data, 5, 'home', on_target = True)[1]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_shots_on_target_against_home'] = get_team_shots_n_matches(team_name, season_data, 10, 'home', on_target = True)[1]
        
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_yellow_cards_per_game_home'] = get_team_cards_per_game(team_name, season_data, 'home', yellow = True)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_yellow_cards_against_per_game_home'] = get_team_cards_per_game(team_name, season_data, 'home', yellow = True)[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_yellow_cards_per_game_away'] = get_team_cards_per_game(team_name, season_data, 'away', yellow = True)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_yellow_cards_against_per_game_away'] = get_team_cards_per_game(team_name, season_data, 'away', yellow = True)[1]
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_red_cards_per_game_home'] = get_team_cards_per_game(team_name, season_data, 'home', yellow = False)[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_red_cards_against_per_game_home'] = get_team_cards_per_game(team_name, season_data, 'home', yellow = False)[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_red_cards_per_game_away'] = get_team_cards_per_game(team_name, season_data, 'away', yellow = False)[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_red_cards_against_per_game_away'] = get_team_cards_per_game(team_name, season_data, 'away', yellow = False)[1]
        
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_xg_per_game_home'] = get_team_xg_per_game(team_name, season_data, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_xg_against_per_game_home'] = get_team_xg_per_game(team_name, season_data, 'home')[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_xg_per_game_away'] = get_team_xg_per_game(team_name, season_data, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_xg_against_per_game_away'] = get_team_xg_per_game(team_name, season_data, 'away')[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_xg_away'] = get_team_xg_n_matches(team_name, season_data, 5, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_xg_away'] = get_team_xg_n_matches(team_name, season_data, 10, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '5_form_xg_against_away'] = get_team_xg_n_matches(team_name, season_data, 5, 'away')[1]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, '10_form_xg_against_away'] = get_team_xg_n_matches(team_name, season_data, 10, 'away')[1]
        
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_xg_home'] = get_team_xg_n_matches(team_name, season_data, 5, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_xg_home'] = get_team_xg_n_matches(team_name, season_data, 10, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '5_form_xg_against_home'] = get_team_xg_n_matches(team_name, season_data, 5, 'home')[1]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, '10_form_xg_against_home'] = get_team_xg_n_matches(team_name, season_data, 10, 'home')[1]
        teams_datasets_seasonal.append(team_1_games)
    merged_teams_datasets_seasonal = teams_datasets_seasonal[0]
    for df in teams_datasets_seasonal[1:]:
        merged_teams_datasets_seasonal = merged_teams_datasets_seasonal.combine_first(df)
    merged_teams_datasets_seasonal.drop(['FTR',	'HS',	'AS',	'HST',	'AST',	'HF',	'AF',	'HC',	'AC',	'HY',	'AY',	'HR',	'AR','BFCA',	'1XBCH',	'Home Team',	'Away Team'],axis= 1, inplace = True, errors = 'ignore')
    return merged_teams_datasets_seasonal


# In[ ]:




