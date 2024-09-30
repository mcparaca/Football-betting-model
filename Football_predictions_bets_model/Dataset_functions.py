#!/usr/bin/env python
# coding: utf-8

# In this file, we consolidate all the functions utilized throughout various stages of this project. This comprehensive repository includes key functions for data gathering, preprocessing, feature engineering, model training, and prediction. By centralizing these functions, we enhance the modularity and maintainability of our codebase, allowing for easier updates and improvements in the future.
# 
# Having all functions in one place also facilitates collaboration and ensures consistency across different notebooks and processes. This structured approach not only streamlines our workflow but also provides a clear reference point for anyone looking to understand or modify the project's functionality.
# 
# Overall, this function library is a critical component of our project, enabling efficient execution and fostering a more organized coding environment.

# In[3]:


import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium.webdriver.support.ui import WebDriverWait
import re 


def clean_seasonal_data(dataset):
    """
    Clean and preprocess the given seasonal dataset.

    This function performs the following operations on the input dataset:
    - Drops unnecessary columns that do not contribute to the analysis.
    - Converts categorical variables 'HomeTeam' and 'AwayTeam' into dummy variables.
    - Ensures boolean columns are converted to integers.
    - Trims the date strings to the first 10 characters and converts them into 
      datetime format.

    Parameters:
    dataset (pandas.DataFrame): The input DataFrame containing seasonal match data.

    Returns:
    pandas.DataFrame: The cleaned and preprocessed DataFrame with unnecessary columns 
                      removed, categorical variables converted to dummy variables, 
                      boolean columns as integers, and date formatted correctly.

    Raises:
    ValueError: If the 'Date' column cannot be converted to datetime format.

    """
    dataset.drop(['HTHG','HTAG','HTR','Referee', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA','AvgH','AvgD',	'AvgA',	'B365>2.5',	'B365<2.5',	'P>2.5',	'P<2.5',	'Max>2.5',	'Max<2.5',	'Avg>2.5',	'Avg<2.5',	'AHh',	'B365AHH',	'B365AHA',	'PAHH',	'PAHA',	'MaxAHH',	'MaxAHA',	'AvgAHH',	'AvgAHA',	'B365CH',	'B365CD',	'B365CA',	'BWCH',	'BWCD',	'BWCA',	'IWCH',	'IWCD',	'IWCA',	'PSCH',	'PSCD',	'PSCA',	'WHCH',	'WHCD',	'WHCA',	'VCCH',	'VCCD',	'VCCA',	'MaxCH',	'MaxCD',	'MaxCA',	'AvgCH',	'AvgCD',	'AvgCA',	'B365C>2.5',	'B365C<2.5',	'PC>2.5',	'PC<2.5',	'MaxC>2.5',	'MaxC<2.5',	'AvgC>2.5',	'AvgC<2.5',	'AHCh',	'B365CAHH',	'B365CAHA',	'PCAHH',	'PCAHA',	'MaxCAHH',	'MaxCAHA',	'AvgCAHH',	'AvgCAHA','BFH',	'BFD',	'BFA',	'1XBH',	'1XBD',	'1XBA',	'BFEH',	'BFED',	'BFEA',	'BFE>2.5',	'BFE<2.5',	'BFEAHH',	'BFEAHA',	'BFCH',	'BFCD',	'BFCA'	'1XBCH',	'1XBCD',	'1XBCA',	'BFECH',	'BFECD',	'BFECA',	'BFEC>2.5',	'BFEC<2.5',	'BFECAHH',	'BFECAHA'	 ], inplace =True, axis = 1, errors='ignore')
    dataset = pd.get_dummies(dataset, columns=['HomeTeam', 'AwayTeam'])
    dummy_columns = dataset.select_dtypes(bool).columns
    dataset[dummy_columns] = dataset[dummy_columns].astype(int)
    dataset['Date'] = dataset['Date'].str[:10]
    try:
        dataset['Date'] = pd.to_datetime(dataset['Date'],format = '%d/%m/%Y') 
    except:
        dataset['Date'] = pd.to_datetime(dataset['Date']) 
    return dataset

def clean_v2(dataset):
    """
    Clean the given dataset by removing unnecessary columns.
    
    This function takes a dataset as input and drops a predefined set of columns that are 
    considered unnecessary for further analysis. The columns removed include various match 
    statistics and betting odds that do not contribute to the predictive modeling.
    
    Parameters:
    dataset (pandas.DataFrame): The input DataFrame containing match data to be cleaned.
    
    Returns:
    pandas.DataFrame: The cleaned DataFrame with specified columns removed.
    
    Raises:
    ValueError: If the DataFrame does not contain the specified columns to drop.
    """
    dataset.drop(['HTHG','HTAG','HTR','Referee', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA','AvgH','AvgD',	'AvgA',	'B365>2.5',	'B365<2.5',	'P>2.5',	'P<2.5',	'Max>2.5',	'Max<2.5',	'Avg>2.5',	'Avg<2.5',	'AHh',	'B365AHH',	'B365AHA',	'PAHH',	'PAHA',	'MaxAHH',	'MaxAHA',	'AvgAHH',	'AvgAHA',	'B365CH',	'B365CD',	'B365CA',	'BWCH',	'BWCD',	'BWCA',	'IWCH',	'IWCD',	'IWCA',	'PSCH',	'PSCD',	'PSCA',	'WHCH',	'WHCD',	'WHCA',	'VCCH',	'VCCD',	'VCCA',	'MaxCH',	'MaxCD',	'MaxCA',	'AvgCH',	'AvgCD',	'AvgCA',	'B365C>2.5',	'B365C<2.5',	'PC>2.5',	'PC<2.5',	'MaxC>2.5',	'MaxC<2.5',	'AvgC>2.5',	'AvgC<2.5',	'AHCh',	'B365CAHH',	'B365CAHA',	'PCAHH',	'PCAHA',	'MaxCAHH',	'MaxCAHA',	'AvgCAHH',	'AvgCAHA','BFH',	'BFD',	'BFA',	'1XBH',	'1XBD',	'1XBA',	'BFEH',	'BFED',	'BFEA',	'BFE>2.5',	'BFE<2.5',	'BFEAHH',	'BFEAHA',	'BFCH',	'BFCD',	'BFCA'	'1XBCH',	'1XBCD',	'1XBCA',	'BFECH',	'BFECD',	'BFECA',	'BFEC>2.5',	'BFEC<2.5',	'BFECAHH',	'BFECAHA'	 ], inplace =True, axis = 1, errors='ignore')
    return dataset
    
def get_team_matches(team_name, dataset):
    """
    Retrieve matches played by a specific team from the dataset.
    
    This function filters the dataset to extract all matches involving a given team. 
    It identifies games where the specified team has a score above 0.5 in either the 
    home or away columns, indicating that the team participated in those matches. 
    Additionally, it creates a new column to represent the matchday number.
    
    Parameters:
    team_name (str): The name of the team for which to retrieve matches.
    dataset (pandas.DataFrame): The input DataFrame containing match data.
    
    Returns:
    pandas.DataFrame: A DataFrame containing all matches played by the specified team, 
                       along with a new column indicating the matchday.
    
    Raises:
    KeyError: If the specified team name does not match any columns in the dataset.
    """
    team_games =  dataset[dataset[dataset.filter(like=team_name).columns].gt(0.5).any(axis=1)]
    matchday_array = np.arange(1, len(team_games) + 1)
    team_games['matchday'] = matchday_array
    return team_games

def get_team_home_matches(team_name, dataset):
    """
    Retrieve home matches played by a specific team from the dataset.
    
    This function filters the dataset to extract all home matches for a given team. 
    It utilizes the `get_team_matches` function to first obtain all matches involving 
    the specified team, then further narrows down the results to only those where the 
    team is playing at home. 
    
    Parameters:
    team_name (str): The name of the team for which to retrieve home matches.
    dataset (pandas.DataFrame): The input DataFrame containing match data.
    
    Returns:
    pandas.DataFrame: A DataFrame containing all home matches played by the specified team.
    
    Raises:
    KeyError: If the specified team name does not match any columns in the dataset.
    """

    team_matches = get_team_matches(team_name, dataset)
    team_home_games =  team_matches[team_matches[team_matches.filter(like=f"HomeTeam_{team_name}").columns].gt(0.5).any(axis=1)]
    return team_home_games

def get_team_away_matches(team_name, dataset):
    """
    Retrieve away matches played by a specific team from the dataset.
    
    This function filters the dataset to extract all away matches for a given team. 
    It utilizes the `get_team_matches` function to first obtain all matches involving 
    the specified team, then further narrows down the results to only those where the 
    team is playing away.
    
    Parameters:
    team_name (str): The name of the team for which to retrieve away matches.
    dataset (pandas.DataFrame): The input DataFrame containing match data.
    
    Returns:
    pandas.DataFrame: A DataFrame containing all away matches played by the specified team.
    
    Raises:
    KeyError: If the specified team name does not match any columns in the dataset.
    """
    team_matches = get_team_matches(team_name, dataset)
    team_away_games =  team_matches[team_matches[team_matches.filter(like=f"AwayTeam_{team_name}").columns].gt(0.5).any(axis=1)]
    return team_away_games

def last_n_matches_form(team_games, team_name, n):
    """
    Calculate the points form of a team over the last N matches.
    
    This function computes the cumulative points earned by a specified team in 
    its last N matches. It evaluates the match results and assigns points based 
    on whether the team won, lost, or drew. The function maintains a running total 
    of points for the specified number of recent matches.
    
    Parameters:
    team_games (pandas.DataFrame): A DataFrame containing match data for the team.
    team_name (str): The name of the team for which to calculate the form.
    n (int): The number of recent matches to consider for the points form.
    
    Returns:
    list: A list containing the cumulative points for the team over the last N matches.
    
    Notes:
    - A win awards 3 points, a draw 1 point, and a loss 0 points.
    - The function initializes with a starting point of 0 and returns a list 
      of cumulative points for each match up to the last N matches.
    
    Example:
        team_form = last_n_matches_form(team_games, 'Team A', 5)
    """ 
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
    """
    Calculate the total points earned by a specified team in all matches, home matches, or away matches.
    
    This function computes the cumulative points for a specified team based on the 
    match results from a given dataset. It allows for calculation across all matches 
    or can be restricted to only home or away matches, as specified by the user. 
    
    Parameters:
    team_name (str): The name of the team for which to calculate the total points.
    dataset (pandas.DataFrame): A DataFrame containing match data for all teams.
    which (str): Specifies the type of matches to consider ('all', 'home', or 'away'). 
                  Default is 'all'.
    
    Returns:
    list: A list containing the cumulative points for the team across the specified matches.
    
    Notes:
    - A win awards 3 points, a draw 1 point, and a loss 0 points.
    - The function initializes with a starting point of 0 and returns a list 
      of cumulative points for each match.
    
    Example:
        total_points = team_current_points_total('Team A', dataset, which='home')
    """

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
    """
    Retrieve the form of a specified team over the last 'n' matches.
    
    This function calculates the performance of a team in terms of points earned over 
    the last 'n' matches, based on the specified match type (home, away, or all).
    
    Parameters:
    team_name (str): The name of the team for which to calculate the form.
    dataset (pandas.DataFrame): A DataFrame containing match data for all teams.
    n (int): The number of last matches to consider for calculating the form.
    which (str): Specifies the type of matches to consider ('all', 'home', or 'away'). 
                  Default is 'all'.
    
    Returns:
    list: A list containing the points accumulated by the team in the last 'n' matches.
    
    Notes:
    - The function uses the helper function `last_n_matches_form` to calculate 
      the points based on match results.
    - A win awards 3 points, a draw 1 point, and a loss 0 points.
    
    Example:
        recent_form = get_team_form('Team A', dataset, n=5, which='away')
    """
    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    form_n_matches = last_n_matches_form(team_games, team_name, n)
    return form_n_matches


def get_team_goals_total(team_name, dataset, which = 'all'):
    """
    Calculate the total goals scored and conceded by a specified team, along with 
    goals per game statistics over a given set of matches.
    
    This function retrieves the total goals scored and conceded by the team, as well 
    as the average goals scored and conceded per game, based on the specified match type 
    (home, away, or all).
    
    Parameters:
    team_name (str): The name of the team for which to calculate goal statistics.
    dataset (pandas.DataFrame): A DataFrame containing match data for all teams.
    which (str): Specifies the type of matches to consider ('all', 'home', or 'away'). 
                  Default is 'all'.
    
    Returns:
    tuple: A tuple containing four lists:
        - Total goals scored by the team in the specified matches.
        - Total goals conceded by the team in the specified matches.
        - Average goals scored per game by the team.
        - Average goals conceded per game by the team.
    
    Notes:
    - The function utilizes helper functions to retrieve matches played by the specified team.
    - Goals are tallied based on whether the team is playing at home or away.
    
    Example:
        goals_data = get_team_goals_total('Team A', dataset, which='home')
    """
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
    """
    Calculate the total goals scored and conceded by a specified team over the last N matches.
    
    This function retrieves the goals scored and goals conceded by the team in their 
    last N matches, allowing for the tracking of recent performance.
    
    Parameters:
    team_games (pandas.DataFrame): A DataFrame containing the matches played by the team.
    team_name (str): The name of the team for which to calculate goals.
    n (int): The number of most recent matches to consider for the calculation.
    
    Returns:
    tuple: A tuple containing two lists:
        - Total goals scored by the team in the last N matches.
        - Total goals conceded by the team in the last N matches.
    
    Notes:
    - The function iterates over the matches and updates the counts based on whether 
      the team is playing at home or away.
    - Only the last N matches are considered for the goals calculation.
    
    Example:
        recent_goals = last_n_matches_goals(team_games, 'Team A', n=5)
    """
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
    """
    Retrieve the total goals scored and conceded by a specified team over the last N matches.
    
    This function allows for the calculation of a team's performance by retrieving the 
    number of goals they have scored and conceded in either home, away, or all matches.
    
    Parameters:
    team_name (str): The name of the team for which to retrieve goals data.
    dataset (pandas.DataFrame): A DataFrame containing the match data for all teams.
    n (int): The number of most recent matches to consider for the calculation.
    which (str): Specifies whether to consider 'home', 'away', or 'all' matches 
                  (default is 'all').
    
    Returns:
    tuple: A tuple containing two lists:
        - Total goals scored by the team in the last N matches.
        - Total goals conceded by the team in the last N matches.
    
    Example:
        goals, against = get_team_goals_n_matches('Team A', dataset, n=5, which='home')
    """

    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    team_goals_n_matches, goals_against_n_matches = last_n_matches_goals(team_games, team_name, n)
    return  team_goals_n_matches, goals_against_n_matches


def get_team_shots_per_game(team_name, dataset, which = 'all', on_target = False):
    """
    Calculate the average number of shots per game for a specified team, as well as the 
    average shots conceded, over a series of matches.
    
    This function allows users to analyze a team's shooting performance, either at home 
    or away, and can focus on total shots or shots on target.
    
    Parameters:
    team_name (str): The name of the team for which to retrieve shooting statistics.
    dataset (pandas.DataFrame): A DataFrame containing the match data for all teams.
    which (str): Specifies whether to consider 'home', 'away', or 'all' matches 
                  (default is 'all').
    on_target (bool): If True, calculates shots on target; if False, calculates total shots 
                      (default is False).
    
    Returns:
    tuple: A tuple containing two lists:
        - Average shots per game for the specified team.
        - Average shots conceded per game by the specified team.
    
    Example:
        shots_per_game, shots_against = get_team_shots_per_game('Team A', dataset, 
                                                                  which='home', 
                                                                  on_target=True)
    """

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
    """
    Calculate the total number of shots taken and shots conceded by a specified team 
    over the last n matches.
    
    This function helps in analyzing a team's recent shooting performance and the 
    defensive effectiveness in terms of shots allowed. The analysis can be done for 
    either total shots or shots on target.
    
    Parameters:
    team_games (pandas.DataFrame): A DataFrame containing the match data for the team.
    team_name (str): The name of the team for which to calculate shooting statistics.
    n (int): The number of most recent matches to consider in the calculation.
    on_target (bool): If True, calculates shots on target; if False, calculates total shots 
                      (default is False).
    
    Returns:
    tuple: A tuple containing two lists:
        - Total shots taken by the specified team over the last n matches.
        - Total shots conceded by the specified team over the last n matches.
    
    Example:
        team_shots, shots_against = last_n_matches_shots(team_games, 'Team A', 5, on_target=True)
    """

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
    """
    Retrieve the total number of shots taken and shots conceded by a specified team 
    over the last n matches, based on the selected match type (home, away, or all).
    
    This function provides insights into a team's shooting performance in terms of 
    total shots or shots on target, depending on the parameters.
    
    Parameters:
    team_name (str): The name of the team for which to calculate shooting statistics.
    dataset (pandas.DataFrame): The dataset containing match information.
    n (int): The number of most recent matches to consider in the calculation.
    which (str): Specifies which matches to include. Options are 'home', 'away', 
                  or 'all' (default is 'all').
    on_target (bool): If True, calculates shots on target; if False, calculates 
                      total shots (default is False).
    
    Returns:
    tuple: A tuple containing two lists:
        - Total shots taken by the specified team over the last n matches.
        - Total shots conceded by the specified team over the last n matches.
    
    Example:
        team_shots, shots_against = get_team_shots_n_matches('Team A', dataset, 5, which='home', on_target=True)
    """

    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    team_shots_n_matches, shots_against_n_matches = last_n_matches_shots(team_games, team_name, n, on_target = on_target)
    return  team_shots_n_matches, shots_against_n_matches


def get_team_cards_per_game(team_name, dataset, which = 'all', yellow = True):
    """
    Calculate the average number of yellow or red cards received by a specified team 
    and the average cards received by their opponents per game, over a given match type 
    (home, away, or all).
    
    This function analyzes disciplinary statistics to evaluate a team's discipline and 
    their opponents' disciplinary tendencies.
    
    Parameters:
    team_name (str): The name of the team for which to calculate card statistics.
    dataset (pandas.DataFrame): The dataset containing match information.
    which (str): Specifies which matches to include. Options are 'home', 'away', 
                  or 'all' (default is 'all').
    yellow (bool): If True, calculates average yellow cards; if False, calculates 
                   average red cards (default is True).
    
    Returns:
    tuple: A tuple containing two lists:
        - Average cards received by the specified team per game.
        - Average cards received by opponents against the specified team per game.
    
    Example:
        team_yellow_cards, opponent_yellow_cards = get_team_cards_per_game('Team A', dataset, which='home', yellow=True)
    """
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

def get_team_corners_per_game(team_name, dataset, which = 'all'):
    """
    Calculate the average number of corners earned by a specified team and the 
    average corners conceded against that team per game, over a given match type 
    (home, away, or all).
    
    This function provides insight into a team's offensive capabilities regarding 
    corner kicks and their defensive performance against corners.
    
    Parameters:
    team_name (str): The name of the team for which to calculate corner statistics.
    dataset (pandas.DataFrame): The dataset containing match information.
    which (str): Specifies which matches to include. Options are 'home', 'away', 
                  or 'all' (default is 'all').
    
    Returns:
    tuple: A tuple containing two lists:
        - Average corners earned by the specified team per game.
        - Average corners conceded against the specified team per game.
    
    Example:
        team_corners, opponent_corners = get_team_corners_per_game('Team A', dataset, which='home')
    """
    team_corners = [0]
    corners_against = [0]
    last_corners =[0]
    last_corners_against = [0]
    team_corners_per_game = [0]
    against_corners_per_game = [0]

    home_corners = 'HC'
    away_corners = 'AC'
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row[f"HomeTeam_{team_name}"] == 1 :
                last_corners.append(row[home_corners])
                last_corners_against.append(row[away_corners])
            elif row[f"AwayTeam_{team_name}"] == 1:
                last_corners.append(row[away_corners])
                last_corners_against.append(row[home_corners])
            team_corners.append(np.sum(last_corners))
            corners_against.append(np.sum(last_corners_against))
            team_corners_per_game.append(team_corners[-1]/len(team_corners))
            against_corners_per_game.append(corners_against[-1]/len(corners_against))
    elif which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_corners.append(row[home_corners])
            last_corners_against.append(row[away_corners])
            team_corners.append(np.sum(last_corners))
            corners_against.append(np.sum(last_corners_against))
            team_corners_per_game.append(team_corners[-1]/len(team_corners))
            against_corners_per_game.append(corners_against[-1]/len(corners_against))
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_corners.append(row[away_corners])
            last_corners_against.append(row[home_corners])
            team_corners.append(np.sum(last_corners))
            corners_against.append(np.sum(last_corners_against))
            team_corners_per_game.append(team_corners[-1]/len(team_corners))
            against_corners_per_game.append(corners_against[-1]/len(corners_against))
    del team_corners[-1]
    del corners_against[-1]
    del team_corners_per_game[-1]
    del against_corners_per_game[-1]
    return team_corners_per_game, against_corners_per_game

def get_team_offsides_per_game(team_name, dataset, which = 'all'):
    """
    Calculate the average number of offsides committed by a specified team and the 
    average offsides conceded against that team per game, over a given match type 
    (home, away, or all).
    
    This function helps to analyze a team's attacking play in terms of offside 
    situations and the defensive pressure they face regarding offsides.
    
    Parameters:
    team_name (str): The name of the team for which to calculate offside statistics.
    dataset (pandas.DataFrame): The dataset containing match information.
    which (str): Specifies which matches to include. Options are 'home', 'away', 
                  or 'all' (default is 'all').
    
    Returns:
    tuple: A tuple containing two lists:
        - Average offsides committed by the specified team per game.
        - Average offsides conceded against the specified team per game.
    
    Example:
        team_offsides, opponent_offsides = get_team_offsides_per_game('Team A', dataset, which='home')
    """

    team_offsides = [0]
    offsides_against = [0]
    last_offsides =[0]
    last_offsides_against = [0]
    team_offsides_per_game = [0]
    against_offsides_per_game = [0]

    home_offsides = 'HO'
    away_offsides = 'AO'
    if which =='all':
        team_games =  get_team_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            if row[f"HomeTeam_{team_name}"] == 1 :
                last_offsides.append(row[home_offsides])
                last_offsides_against.append(row[away_offsides])
            elif row[f"AwayTeam_{team_name}"] == 1:
                last_offsides.append(row[away_offsides])
                last_offsides_against.append(row[home_offsides])
            team_offsides.append(np.sum(last_offsides))
            offsides_against.append(np.sum(last_offsides_against))
            team_offsides_per_game.append(team_offsides[-1]/len(team_offsides))
            against_offsides_per_game.append(offsides_against[-1]/len(offsides_against))
    elif which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_offsides.append(row[home_offsides])
            last_offsides_against.append(row[away_offsides])
            team_offsides.append(np.sum(last_offsides))
            offsides_against.append(np.sum(last_offsides_against))
            team_offsides_per_game.append(team_offsides[-1]/len(team_offsides))
            against_offsides_per_game.append(offsides_against[-1]/len(offsides_against))
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)
        for index, row in team_games.iterrows():
            last_offsides.append(row[away_offsides])
            last_offsides_against.append(row[home_offsides])
            team_offsides.append(np.sum(last_offsides))
            offsides_against.append(np.sum(last_offsides_against))
            team_offsidess_per_game.append(team_offsides[-1]/len(team_offsides))
            against_offsides_per_game.append(offsides_against[-1]/len(offsides_against))
    del team_offsides[-1]
    del offsides_against[-1]
    del team_offsides_per_game[-1]
    del against_offsides_per_game[-1]
    return team_offsides_per_game, against_offsides_per_game



def get_team_xg_per_game(team_name, dataset, which = 'all'):
    """
    Calculate the average expected goals (xG) generated by a specified team 
    and the average expected goals conceded against that team per game, based on 
    home or away matches.
    
    Expected goals is a metric used to assess the quality of scoring chances 
    and is a valuable statistic for analyzing team performance in terms of 
    attacking efficiency and defensive resilience.
    
    Parameters:
    team_name (str): The name of the team for which to calculate xG statistics.
    dataset (pandas.DataFrame): The dataset containing match information.
    which (str): Specifies which matches to include. Options are 'home', 'away', 
                  or 'all' (default is 'all').
    
    Returns:
    tuple: A tuple containing two lists:
        - Average expected goals generated by the specified team per game.
        - Average expected goals conceded against the specified team per game.
    
    Example:
        team_xg, opponent_xg = get_team_xg_per_game('Team A', dataset, which='home')
    """

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
    """
    Calculate the expected goals (xG) generated and conceded by a specified 
    team over the last n matches, based on home and away performances.
    
    This function is useful for assessing a team's recent attacking output 
    and defensive vulnerabilities in terms of expected goals, providing insight 
    into their performance trends.
    
    Parameters:
    team_games (pandas.DataFrame): A DataFrame containing match information 
                                     for the specified team.
    team_name (str): The name of the team for which to calculate xG statistics.
    n (int): The number of recent matches to consider for the calculation.
    
    Returns:
    tuple: A tuple containing two lists:
        - Total expected goals generated by the specified team over the last n 
          matches.
        - Total expected goals conceded against the specified team over the last n 
          matches.
    
    Example:
        team_xg_last_n, opponent_xg_last_n = last_n_matches_xg(team_games, 'Team A', n=5)
    """

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
    """
    Retrieve the expected goals (xG) generated and conceded by a specified 
    team over the last n matches, based on home and away performances.
    
    This function aggregates the team's xG statistics to provide insight into 
    their recent attacking and defensive effectiveness.
    
    Parameters:
    team_name (str): The name of the team for which to retrieve xG statistics.
    dataset (pandas.DataFrame): A DataFrame containing match information for 
                                 all teams.
    n (int): The number of recent matches to consider for the calculation.
    which (str): Specifies the match context: 
                  'all' for all matches, 
                  'home' for home matches, 
                  'away' for away matches.
    
    Returns:
    tuple: A tuple containing two lists:
        - Expected goals generated by the specified team over the last n matches.
        - Expected goals conceded against the specified team over the last n matches.
    
    Example:
        team_xg_n, opponent_xg_n = get_team_xg_n_matches('Team A', dataset, n=5)
    """

    if which =='home':
        team_games =  get_team_home_matches(team_name, dataset)
    elif which =='away':
        team_games =  get_team_away_matches(team_name, dataset)   
    elif which =='all':
        team_games =  get_team_matches(team_name, dataset) 
    team_shots_n_matches, shots_against_n_matches = last_n_matches_xg(team_games, team_name, n)
    return  team_shots_n_matches, shots_against_n_matches



import requests
from bs4 import BeautifulSoup

# URL of the website to scrape




def scrape_data(url, season = '2023-2024', league_identifier = '9'):
    """
    Scrape expected goals (xG) data for football matches from a specified URL. 
    The function retrieves match details for a given season and league.
    
    Parameters:
    url (str): The URL from which to scrape the match data.
    season (str): The season for which to retrieve match data (default is '2023-2024').
    league_identifier (str): The identifier for the league (default is '9').
    
    Returns:
    pandas.DataFrame: A DataFrame containing the scraped match data with the following columns:
        - Date: The date of the match.
        - Home Team: The name of the home team.
        - Away Team: The name of the away team.
        - Home xG: The expected goals for the home team.
        - Away xG: The expected goals for the away team.
    
    Example:
        xgoals_data = scrape_data('https://example.com/matches')
    """

    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # Find the table with match data
        table = soup.find('table', {'id': f'sched_{season}_{league_identifier}_1'})
    
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

def join_scraped(url, season_data, season, teams, league_identifier):
    """
    Join scraped expected goals (xG) data with the existing season data for football matches.
    
    This function fetches xG data from a specified URL, cleans it, and merges it with the provided 
    season data based on home and away teams.
    
    Parameters:
    - url (str): The URL from which to scrape the xG data.
    - season_data (pandas.DataFrame): The DataFrame containing the current season match data.
    - season (str): The season for which data is being processed.
    - teams (list): A list of team names in the order corresponding to their identifiers.
    - league_identifier (str): The identifier for the league.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the updated season data with added xG information 
      for both home and away teams.
    
    Example:
        updated_season_data = join_scraped('https://example.com/matches', season_data, '2023-2024', teams, '9')
    """

    xgoals= scrape_data(url, season, league_identifier)
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
    
def get_team_ovr25_per_game(team_name, dataset):
    """
    Calculate the average number of matches with over 2.5 goals per game for a specified football team.
    
    This function analyzes the match data for a given team and computes the average number of 
    matches in which the total goals scored (home and away) exceeds 2.5.
    
    Parameters:
    - team_name (str): The name of the football team for which to calculate the average over 2.5 goals per game.
    - dataset (pandas.DataFrame): The DataFrame containing match data including goals scored.
    
    Returns:
    - list: A list of average values indicating the proportion of games with over 2.5 goals 
      per match played by the specified team.
    
    Example:
        avg_over_2_5 = get_team_ovr25_per_game('Team A', season_data)
    """

    team_ovr = [0]
    last_ovr =[0]
    team_ovr_per_game = [0]
    team_games =  get_team_matches(team_name, dataset)
    team_games['over_2_5_goals'] = (team_games['FTHG'] + team_games['FTAG']) > 2.5
    for index, row in team_games.iterrows():
        if row[f"HomeTeam_{team_name}"] == 1 :
            last_ovr.append(row['over_2_5_goals'])

        elif row[f"AwayTeam_{team_name}"] == 1:
            last_ovr.append(row['over_2_5_goals'])

        team_ovr.append(np.sum(last_ovr))
        team_ovr_per_game.append(team_ovr[-1]/(len(team_ovr)-1))
    del team_ovr[-1]
    del team_ovr_per_game[-1]
    return team_ovr_per_game

def create_seasonal_table(season_data, teams):
    """
    Create a seasonal statistics table for each team in a given dataset.
    
    This function generates a comprehensive DataFrame containing various performance metrics 
    for each specified team over the course of a season. It calculates statistics such as 
    goal form, corners, expected goals (xG), and more based on home and away games.
    
    Parameters:
    - season_data (pandas.DataFrame): The DataFrame containing match data for the season, 
      including team performance and match results.
    - teams (list of str): A list of team names for which to generate the seasonal statistics.
    
    Returns:
    - pandas.DataFrame: A merged DataFrame containing the seasonal statistics for all teams, 
      with individual team data combined into a single table.
    
    Example:
        seasonal_table = create_seasonal_table(season_data, ['Team A', 'Team B', 'Team C'])
    """

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

        
        team_1_games['ovr25_per_game'] = get_team_ovr25_per_game(team_name, season_data )
        team_1_games['ovr25_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['ovr25_per_game'], np.nan)
        team_1_games['ovr25_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['ovr25_per_game'], np.nan)
        
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

        team_1_games['total_corners_per_game'] = get_team_corners_per_game(team_name, season_data, 'all')[0]
        team_1_games['total_corners_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_corners_per_game'], np.nan)
        team_1_games['total_corners_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_corners_per_game'], np.nan)
        
        
        team_1_games['total_corners_against_per_game'] = get_team_corners_per_game(team_name, season_data, 'all')[1]
        team_1_games['total_corners_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_corners_against_per_game'], np.nan)
        team_1_games['total_corners_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_corners_against_per_game'], np.nan)

        # team_1_games['total_offsides_per_game'] = get_team_offsides_per_game(team_name, season_data, 'all')[0]
        # team_1_games['total_offsides_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_offsides_per_game'], np.nan)
        # team_1_games['total_offsides_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_offsides_per_game'], np.nan)
        
        
        # team_1_games['total_offsides_against_per_game'] = get_team_offsides_per_game(team_name, season_data, 'all')[1]
        # team_1_games['total_offsides_against_per_game_home_team'] = np.where(team_1_games[f"HomeTeam_{team_name}"] == 1, team_1_games['total_offsides_against_per_game'], np.nan)
        # team_1_games['total_offsides_against_per_game_away_team'] = np.where(team_1_games[f"AwayTeam_{team_name}"] == 1, team_1_games['total_offsides_against_per_game'], np.nan)

        
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

        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_corners_per_game_home'] = get_team_corners_per_game(team_name, season_data, 'home')[0]
        team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_corners_against_per_game_home'] = get_team_corners_per_game(team_name, season_data, 'home')[1]
        
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_corners_per_game_away'] = get_team_corners_per_game(team_name, season_data, 'away')[0]
        team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_corners_against_per_game_away'] = get_team_corners_per_game(team_name, season_data, 'away')[1]


        # team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_offsides_per_game_home'] = get_team_offsides_per_game(team_name, season_data, 'home')[0]
        # team_1_games.loc[team_1_games[f'HomeTeam_{team_name}'] == 1, 'total_offsides_against_per_game_home'] = get_team_offsides_per_game(team_name, season_data, 'home')[1]
        
        # team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_offsides_per_game_away'] = get_team_offsides_per_game(team_name, season_data, 'away')[0]
        # team_1_games.loc[team_1_games[f'AwayTeam_{team_name}'] == 1, 'total_offsides_against_per_game_away'] = get_team_offsides_per_game(team_name, season_data, 'away')[1]

        
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
    merged_teams_datasets_seasonal.drop(['FTR',	'HS',	'AS',	'HST',	'AST',	'HF',	'AF',	'HY',	'AY',	'HR',	'AR','BFCA',	'1XBCH',	'Home Team',	'Away Team'],axis= 1, inplace = True, errors = 'ignore')
    return merged_teams_datasets_seasonal


def prepare_dataframe(csv_file, season):
    """
    Prepare and clean a DataFrame for a specific football season.
    
    This function loads match data from a CSV file, scrapes additional data from 
    an external website, cleans the data, and generates a comprehensive seasonal 
    statistics table for teams. It processes the data to include performance metrics 
    such as goals, corners, and expected goals (xG) for analysis and modeling.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing the match data for the season.
    - season (str): The season identifier (e.g., '2023-2024') used to scrape the corresponding 
      match fixtures and statistics.
    
    Returns:
    - pandas.DataFrame: A cleaned and enriched DataFrame containing seasonal statistics 
      for all teams, ready for analysis or modeling.
    
    Example:
        prepared_data = prepare_dataframe('season_data.csv', '2023-2024')
    """
    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
    league_identifier = '9'
    season_data = join_scraped(url, season_data, season, teams, league_identifier)
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data
    
def calculate_odds(season_data):
    """
    Calculate and normalize betting odds for home wins, draws, and away wins.
    
    This function computes the average betting odds from multiple bookmakers for 
    home wins, draws, and away wins. It then normalizes these odds to account for 
    the bookmaker margin, ensuring they sum to one. The cleaned odds are added 
    to the provided DataFrame, and unnecessary columns are dropped to streamline 
    the data for further analysis.
    
    Parameters:
    - season_data (pandas.DataFrame): A DataFrame containing match data and 
      betting odds from various bookmakers. It should include columns for 
      home win odds, draw odds, and away win odds from different bookmakers.
    
    Returns:
    - pandas.DataFrame: The updated DataFrame with normalized odds for home wins, 
      draws, and away wins, alongside the original match data (minus dropped columns).
    
    Example:
        updated_data = calculate_odds(season_data)
    """

    try:
        season_data['home_win_odds'] = season_data[['B365H', 'BWH', 'IWH', 'PSH']].mean(axis=1)
        season_data['draw_odds'] = season_data[['B365D', 'BWD', 'IWD', 'PSD']].mean(axis=1)
        season_data['away_win_odds'] = season_data[['B365A', 'BWA', 'IWA', 'PSA']].mean(axis=1) 
    except:
        season_data['home_win_odds'] = season_data[['B365H', 'BWH', 'PSH']].mean(axis=1)
        season_data['draw_odds'] = season_data[['B365D', 'BWD', 'PSD']].mean(axis=1)
        season_data['away_win_odds'] = season_data[['B365A', 'BWA', 'PSA']].mean(axis=1) 
    bookmaker_margin = []
    for index, row in season_data.iterrows():
        book_mar = (1/row['home_win_odds'] + 1/row['draw_odds'] + 1/row['away_win_odds'])
        home_odds = 1/row['home_win_odds']/book_mar
        draw_odds = 1/row['draw_odds']/book_mar
        away_odds = 1/row['away_win_odds']/book_mar
        season_data.at[index, 'home_win_odds']= home_odds
        season_data.at[index, 'draw_odds']= draw_odds
        season_data.at[index, 'away_win_odds'] = away_odds
    season_data.drop(['B365H', 'BWH', 'IWH', 'PSH','B365D', 'BWD', 'IWD', 'PSD','B365A', 'BWA', 'IWA', 'PSA','LBH',	'LBD',	'LBA',	'Bb1X2',	'BbMxH',	'BbAvH',	'BbMxD',	'BbAvD',	'BbMxA',	'BbAvA',	'BbOU',	'BbMx>2.5',	'BbAv>2.5',	'BbMx<2.5',	'BbAv<2.5',	'BbAH',	'BbAHh',	'BbMxAHH',	'BbAvAHH',	'BbMxAHA',	'BbAvAHA'], axis = 1, inplace= True, errors = 'ignore')
    return season_data

def get_team_h2h(league_data):
    """
    Calculate head-to-head (H2H) statistics for teams in a league.
    
    This function computes cumulative goals and points for home and away teams 
    when they face each other, storing these statistics in the provided league 
    data DataFrame. The function iterates over each distinct pair of teams, 
    collecting their goals and points from their matches, and calculates 
    the average goals and points for each team in their head-to-head encounters.
    
    Parameters:
    - league_data (pandas.DataFrame): A DataFrame containing match data for 
      a specific league. It should include columns for 'HomeTeam', 'AwayTeam', 
      'FTHG' (full-time home goals), and 'FTAG' (full-time away goals).
    
    Returns:
    - pandas.DataFrame: The updated league_data DataFrame with additional 
      columns for cumulative head-to-head goals and points for both home 
      and away teams in each match.
    
    Example:
        updated_league_data = get_team_h2h(league_data)
    """

    league_data['Home_h2h_Goals'] = np.zeros(len(league_data))
    league_data['Home_h2h_Points'] = np.zeros(len(league_data))
    league_data['Away_h2h_Goals'] = np.zeros(len(league_data))
    league_data['Away_h2h_Points'] = np.zeros(len(league_data))
    teams = league_data['HomeTeam'].unique()
    teams_pairs = [[i,j]  for i in teams for j in teams if i!=j]
    teams_pairs = [sorted([i, j]) for i in teams for j in teams if i != j]

    # Remove duplicates where the same pair appears in reverse order
    distinct_teams_pairs = []
    for pair in teams_pairs:
        if pair not in distinct_teams_pairs:
            distinct_teams_pairs.append(pair)
    teams_pairs = distinct_teams_pairs
    for k in range(len(teams_pairs)):
        team_stats = {}
        teams_h2h = league_data.loc[((league_data['HomeTeam'] == teams_pairs[k][0]) &(league_data['AwayTeam'] == teams_pairs[k][1])) | ((league_data['AwayTeam'] == teams_pairs[k][0]) &(league_data['HomeTeam'] == teams_pairs[k][1]))]
        
        def get_team_stats(team):
            if team not in team_stats:
                team_stats[team] = {'goals': 0, 'points': 0}
            return team_stats[team]['goals'], team_stats[team]['points']
        
        # Function to update the cumulative goals and points for a team
        def update_team_stats(team, goals, points):
            if team not in team_stats:
                team_stats[team] = {'goals': 0, 'points': 0}
            team_stats[team]['goals'] += goals
            team_stats[team]['points'] += points
        
        # Lists to store cumulative data before each match
        home_team_cum_goals = []
        home_team_cum_points = []
        away_team_cum_goals = []
        away_team_cum_points = []
        
        
        for _, row in teams_h2h.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Get cumulative stats for both teams before the match
            home_goals_before, home_points_before = get_team_stats(home_team)
            away_goals_before, away_points_before = get_team_stats(away_team)
            
            # Append the current cumulative stats to the lists
            home_team_cum_goals.append(home_goals_before)
            home_team_cum_points.append(home_points_before)
            away_team_cum_goals.append(away_goals_before)
            away_team_cum_points.append(away_points_before)
            
            # Update the stats after the match
            # Goals
            home_goals = row['FTHG']
            away_goals = row['FTAG']
            update_team_stats(home_team, home_goals, 0)
            update_team_stats(away_team, away_goals, 0)
            
            # Points
            if home_goals > away_goals:
                update_team_stats(home_team, 0, 3)  # Home team wins
            elif home_goals < away_goals:
                update_team_stats(away_team, 0, 3)  # Away team wins
            else:
                update_team_stats(home_team, 0, 1)  # Draw
                update_team_stats(away_team, 0, 1)  # Draw
        # print(home_team_cum_goals)
        # fixtures = np.linspace(0,len(home_team_cum_goals)-1,len(home_team_cum_goals))
        teams_h2h['Home_h2h_Goals'] = home_team_cum_goals/np.linspace(0,len(home_team_cum_goals)-1,len(home_team_cum_goals))
        teams_h2h['Home_h2h_Points'] = home_team_cum_points/np.linspace(0,len(home_team_cum_goals)-1,len(home_team_cum_goals))
        teams_h2h['Away_h2h_Goals'] = away_team_cum_goals/np.linspace(0,len(home_team_cum_goals)-1,len(home_team_cum_goals))
        teams_h2h['Away_h2h_Points'] = away_team_cum_points/np.linspace(0,len(home_team_cum_goals)-1,len(home_team_cum_goals))
        teams_h2h.fillna(0,inplace = True)
        league_data.update(teams_h2h)
    league_data.replace([np.inf, -np.inf], 0, inplace=True)
        # league_data = pd.merge(league_data, teams_h2h[['Date',	'HomeTeam',	'AwayTeam','Home_h2h_Goals', 'Home_h2h_Points', 'Away_h2h_Goals', 'Away_h2h_Points']], on =['Date', 'HomeTeam', 'AwayTeam'], how = 'outer')
    return league_data

def prepare_dataframe_championship(csv_file, season):
    """
    Prepare the seasonal dataset for the Championship league.
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in the Championship league.
    
    Example:
        season_data = prepare_dataframe_championship("championship_data.csv", "2023-2024")
    """
    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/10/{season}/schedule/{season}-Championship-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '10')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data

def prepare_dataframe_spanish1(csv_file, season):
    """
    Prepare the seasonal dataset for La Liga (Spanish Primera Division).
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in La Liga.
    
    Example:
        season_data = prepare_dataframe_spanish1("la_liga_data.csv", "2023-2024")
    """

    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/12/{season}/schedule/{season}-La-Liga-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '12')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data


def prepare_dataframe_spanish2(csv_file, season):
    """
    Prepare the seasonal dataset for the Spanish Segunda Division.
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in the Segunda Division.
    
    Example:
        season_data = prepare_dataframe_spanish2("segunda_division_data.csv", "2023-2024")
    """

    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/17/{season}/schedule/{season}-Segunda-Division-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '17')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data

def prepare_dataframe_italian1(csv_file, season):
    """
    Prepare the seasonal dataset for Italian Serie A.
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in Serie A.
    
    Example:
        season_data = prepare_dataframe_italian1("serie_a_data.csv", "2023-2024")
    """

    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/11/{season}/schedule/{season}-Serie-A-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '11')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data

def prepare_dataframe_italian2(csv_file, season):
    """
    Prepare the seasonal dataset for Italian Serie B.
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in Serie B.
    
    Example:
        season_data = prepare_dataframe_italian2("serie_b_data.csv", "2023-2024")
    """

    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/18/{season}/schedule/{season}-Serie-B-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '18')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data

def prepare_dataframe_german1(csv_file, season):
    """
    Prepare the seasonal dataset for German Bundesliga.
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in the Bundesliga.
    
    Example:
        season_data = prepare_dataframe_german1("bundesliga_data.csv", "2023-2024")
    """

    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/20/{season}/schedule/{season}-Bundesliga-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '20')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data

def prepare_dataframe_german2(csv_file, season):
    """
    Prepare the seasonal dataset for German 2. Bundesliga.
    
    This function reads match data from a CSV file, scrapes additional data 
    from an online source for the specified season, and processes it to 
    create a comprehensive dataset for analysis. It performs the following steps:
    1. Reads the match data from the provided CSV file.
    2. Extracts unique team names from the dataset.
    3. Scrapes additional match statistics from the fbref website.
    4. Cleans the scraped data.
    5. Creates a seasonal table with various statistics for each team.
    6. Calculates betting odds based on the provided match data.
    
    Parameters:
    - csv_file (str): The path to the CSV file containing historical match data.
    - season (str): The season for which the data should be prepared, formatted as 'YYYY-YYYY'.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data for the 
      specified season in the 2. Bundesliga.
    
    Example:
        season_data = prepare_dataframe_german2("bundesliga2_data.csv", "2023-2024")
    """

    season_data = pd.read_csv(csv_file)
    teams = season_data['HomeTeam'].unique()
    url = f"https://fbref.com/en/comps/33/{season}/schedule/{season}-2Bundesliga-Scores-and-Fixtures"
    season_data = join_scraped(url, season_data, season, teams, '33')
    season_data = clean_v2(season_data)
    season_data = create_seasonal_table(season_data,teams)
    season_data = calculate_odds(season_data)
    return season_data

def convert_odds(season_data):
    """
    Convert bookmaker odds into implied probabilities.
    
    This function takes a DataFrame containing betting odds and converts them 
    to implied probabilities by calculating the bookmaker margin. The following 
    odds are converted: home win, draw, and away win. The function updates 
    the DataFrame in place with the converted odds.
    
    Parameters:
    - season_data (pandas.DataFrame): A DataFrame containing the match data 
      with the following columns: 'home_win_odds', 'draw_odds', 
      and 'away_win_odds'.
    
    Returns:
    - pandas.DataFrame: The updated DataFrame with the converted odds 
      reflecting the implied probabilities.
    
    Example:
        updated_season_data = convert_odds(season_data)
    """

    bookmaker_margin = []
    for index, row in season_data.iterrows():
        book_mar = (1/row['home_win_odds'] + 1/row['draw_odds'] + 1/row['away_win_odds'])
        home_odds = 1/row['home_win_odds']/book_mar
        draw_odds = 1/row['draw_odds']/book_mar
        away_odds = 1/row['away_win_odds']/book_mar
        season_data.at[index, 'home_win_odds']= home_odds
        season_data.at[index, 'draw_odds']= draw_odds
        season_data.at[index, 'away_win_odds'] = away_odds
    return season_data

def append_last_element(lst):
    """
    Append the last element of a list to itself.
    
    This function takes a list as input and appends its last element 
    to the end of the list if the list is not empty. If the list is empty, 
    it returns the list unchanged.
    
    Parameters:
    - lst (list): The list to which the last element will be appended.
    
    Returns:
    - list: The modified list with the last element appended, or the 
      original list if it was empty.
    
    Example:
        my_list = [1, 2, 3]
        updated_list = append_last_element(my_list)
        # updated_list will be [1, 2, 3, 3]
    """

    if lst:  # Checks if the list is not empty
        lst.append(lst[-1])  # Appends the last element of the list to itself
    return lst

def scrape_next_fixtures(league_url):
    """
    Scrapes the next fixtures from a specified league webpage.
    
    This function uses Selenium to automate a web browser (Microsoft Edge) 
    to navigate to the provided league URL and scrape the HTML content of the page. 
    It also handles the acceptance of an age verification dialog.
    
    Parameters:
    - league_url (str): The URL of the league page to scrape fixtures from.
    
    Returns:
    - BeautifulSoup object: A BeautifulSoup object containing the parsed HTML 
      of the league page, which can be used to extract fixture information.
    
    Example:
        league_url = "https://example.com/league-fixtures"
        fixtures_soup = scrape_next_fixtures(league_url)
        # Use fixtures_soup to extract next fixtures data
    
    Notes:
    - Ensure that the Edge WebDriver executable is available at the specified path.
    - The function operates in headless mode for efficiency; to see the browser in action, 
      remove the headless option.
    - The function includes a sleep of 5 seconds to allow the page to load completely; 
      adjust this duration as necessary based on your internet speed and the page load time.
    """

    edge_options = Options()
    edge_options.add_argument("--headless")  # Run Edge in headless mode
    edge_options.add_argument("disable-gpu")
    edge_options.add_argument("--log-level=3")  # Suppress logs (optional)
    
    edge_driver_path = 'msedgedriver.exe'  # Path to your Edge WebDriver
    
    service = Service(executable_path=edge_driver_path)
    driver = webdriver.Edge(service=service)
    driver.get(league_url)
    driver.maximize_window()
    time.sleep(5)

    age_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Mam przynajmniej 18 lat i wchodzę')]")
    age_button.click()

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.quit()
    return soup

def create_df_next_fixtures(soup, number_of_matches=10):
    """
    Creates a DataFrame of the next fixtures for a league from the scraped HTML content.
    
    This function extracts relevant information from the BeautifulSoup object representing
    the league's webpage and constructs a DataFrame containing details about upcoming matches,
    including teams, dates, and betting odds.
    
    Parameters:
    - soup (BeautifulSoup): A BeautifulSoup object containing the parsed HTML of the league page.
    - number_of_matches (int, optional): The number of upcoming matches to include in the DataFrame.
      Default is 10.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the next match details, including:
      - 'Date': The date of the match.
      - 'HomeTeam': The name of the home team.
      - 'AwayTeam': The name of the away team.
      - 'home_win_odds': Odds for the home team to win.
      - 'draw_odds': Odds for a draw.
      - 'away_win_odds': Odds for the away team to win.
      - 'Avg<2.5': Odds for the total goals to be under 2.5.
      - 'Avg>2.5': Odds for the total goals to be over 2.5.
    
    Example:
        league_soup = scrape_next_fixtures("https://example.com/league-fixtures")
        fixtures_df = create_df_next_fixtures(league_soup)
        print(fixtures_df)
    
    Notes:
    - The function assumes that the HTML structure of the page matches the expected format,
      specifically that match information is contained within a section with class 'mb-6'.
    - The date format is expected to be day-month-year (e.g., '01.01.2024').
    - Odds are extracted from buttons within the match section; the function assumes that
      odds are presented in a consistent order.
    - The function uses the helper function `append_last_element` to ensure that dates
      match the number of home teams extracted.
    """

    first_match_section = soup.find('section', class_='mb-6')
    home_win_odds = []
    draw_odds = []
    away_win_odds = []
    under_2_5 = []
    over_2_5 = []
    home_teams = []
    away_teams = []
    dates = []
    
    odds = first_match_section.find_all('button')
    for i in range(2,len(odds)):
        if i%5==2:
            home_win_odds.append(float(odds[i].get_text()))
        elif i%5==3:
            draw_odds.append(float(odds[i].get_text()))
        elif i%5==4:
            away_win_odds.append(float(odds[i].get_text()))
        elif i%5==0:
            under_2_5.append(float(odds[i].get_text()))
        elif i%5==1:
            over_2_5.append(float(odds[i].get_text()))
            
    matches = first_match_section.find_all('header')
    for j in range(len(matches)):
        if '1X2' in matches[j].get_text():
            dates.append(matches[j].get_text().split('1X2')[0].split(' ')[1]+'.2024')
        if ' - ' in matches[j].get_text():
            teams = matches[j].get_text()
            home_team = teams.split(' - ')[0]
            away_team = teams.split(' - ')[1]
            home_teams.append(home_team)
            away_teams.append(away_team)
        if len(dates)< len(home_teams):
            append_last_element(dates) 
    data_next_fixtures = {
    'Date': dates[:number_of_matches],
    'HomeTeam': home_teams[:number_of_matches],
    'AwayTeam': away_teams[:number_of_matches],
    'home_win_odds': home_win_odds[:number_of_matches],
    'draw_odds': draw_odds[:number_of_matches],
    'away_win_odds': away_win_odds[:number_of_matches],
    'Avg<2.5': under_2_5[:number_of_matches],
    'Avg>2.5': over_2_5[:number_of_matches]
    }
    next_fixtures = pd.DataFrame(data_next_fixtures)
    next_fixtures['Date'] = pd.to_datetime(next_fixtures['Date'], format = '%d.%m.%Y')
    return next_fixtures

def append_to_df_all(next_fixtures,update_dict, csv_name, new_csv_name):
    """
    Appends new fixture data to an existing matches DataFrame and saves it as a new CSV file.
    
    This function reads the existing matches from a CSV file, updates the fixture information
    with new data, and then saves the updated DataFrame into a new CSV file. It handles 
    the conversion of date formats and fills missing values as needed.
    
    Parameters:
    - next_fixtures (pandas.DataFrame): A DataFrame containing the next match details, including
      home and away teams, match dates, and betting odds.
    - update_dict (dict): A dictionary mapping old team names to updated team names for replacement
      in the DataFrame.
    - csv_name (str): The file name of the existing CSV containing past match data.
    - new_csv_name (str): The file name for the new CSV to save the updated match data.
    
    Returns:
    - None: The function directly saves the updated DataFrame to a new CSV file.
    
    Example:
        next_fixtures_df = create_df_next_fixtures(soup)
        team_updates = {'Old Team Name': 'New Team Name'}
        append_to_df_all(next_fixtures_df, team_updates, 'matches.csv', 'updated_matches.csv')
    
    Notes:
    - The function expects the 'Date' column in the existing CSV to be formatted as 'day/month/year'.
    - It uses forward filling to propagate last valid observation forward to next valid.
    - The betting odds for home wins, draws, and away wins are renamed to match expected columns in the output.
    - The original 'home_win_odds', 'draw_odds', and 'away_win_odds' columns from the new fixtures are dropped after renaming.
    """

    all_matches = pd.read_csv(csv_name)
    try:
        all_matches['Date'] = pd.to_datetime(all_matches['Date'], format = '%d/%m/%Y')
    except:
        all_matches['Date'] = pd.to_datetime(all_matches['Date'])
    next_fixtures[['B365H', 'B365D', 'B365A']] = next_fixtures[['home_win_odds', 'draw_odds', 'away_win_odds']] 
    next_fixtures[['BWH', 'BWD', 'BWA']] = next_fixtures[['home_win_odds', 'draw_odds', 'away_win_odds']] 
    next_fixtures[['PSH', 'PSD', 'PSA']] = next_fixtures[['home_win_odds', 'draw_odds', 'away_win_odds']] 
    next_fixtures.drop(['home_win_odds', 'draw_odds', 'away_win_odds'], axis = 1, inplace = True)
    all_matches = pd.concat([all_matches,next_fixtures])
    all_matches.ffill(inplace = True)
    all_matches['HomeTeam'] = all_matches['HomeTeam'].replace(update_dict)
    all_matches['AwayTeam'] = all_matches['AwayTeam'].replace(update_dict)
    all_matches.to_csv(new_csv_name, index = False)


def update_matches(league_url, file_name):
    """
    Updates the local match data file by downloading the latest data from a given URL.
    
    This function retrieves match data from a specified league URL and saves it to a local file.
    It effectively updates the existing file with the latest data, overwriting any previous content.
    
    Parameters:
    - league_url (str): The URL from which to download the latest match data. This should be 
      a direct link to a file or endpoint that returns the match data in a readable format.
    - file_name (str): The name of the local file where the downloaded data will be saved. 
      If a file with this name already exists, it will be overwritten.
    
    Returns:
    - None: The function does not return any value but saves the downloaded data to a file.
    
    Example:
        update_matches("https://example.com/match_data.csv", "latest_matches.csv")
    
    Notes:
    - Ensure that the URL provided is accessible and returns the correct data format expected 
      in the local file.
    - This function requires the `requests` library to be installed in your environment.
    """

    response = requests.get(league_url)
    with open(file_name, 'wb') as file:
        file.write(response.content)


def get_elo_for_team_on_date(team, match_date, df_elo):
    """
    Retrieves the Elo rating for a specified team on a given match date.
    
    This function filters the Elo ratings DataFrame to find the rating of a specified team 
    on a specific date. It looks for the row in the DataFrame where the match date falls 
    within the 'From' and 'To' date range for that team.
    
    Parameters:
    - team (str): The name of the team for which the Elo rating is being queried.
    - match_date (datetime): The date for which the Elo rating is requested. This should be 
      a valid datetime object.
    - df_elo (pd.DataFrame): A DataFrame containing Elo ratings with columns 'Club', 
      'From', 'To', and 'Elo'. Each row should represent a different time period for a team.
    
    Returns:
    - float or None: The Elo rating for the specified team on the given date. 
      If no rating is found for that date, the function returns None.
    
    Example:
        elo_rating = get_elo_for_team_on_date("Team A", pd.to_datetime("2024-09-30"), df_elo)
    
    Notes:
    - Ensure that the match date is in the same format as the dates in the 'From' and 'To' 
      columns of the DataFrame.
    - The function assumes the DataFrame is properly formatted and includes the necessary 
      columns for filtering.
    """

    # Filter by team
    team_elo_data = df_elo[df_elo['Club'] == team]
    # Find the row where the match date is between 'From' and 'To'
    elo_row = team_elo_data[(team_elo_data['From'] <= match_date) & (team_elo_data['To'] >= match_date)]
    if not elo_row.empty:
        return elo_row['Elo'].values[0]
    return None
    
def find_elo_from_team_df(team, match_date, team_dfs):
    """
    Retrieves the Elo rating for a specified team on a given match date from a dictionary of team DataFrames.
    
    This function checks if the specified team has an associated DataFrame in the provided 
    dictionary. If it does, the function filters the DataFrame to find the Elo rating for 
    the team that corresponds to the given match date. The function returns the Elo rating 
    if it exists within the specified date range.
    
    Parameters:
    - team (str): The name of the team for which the Elo rating is being queried.
    - match_date (datetime): The date for which the Elo rating is requested. This should be 
      a valid datetime object.
    - team_dfs (dict): A dictionary where keys are team names and values are DataFrames 
      containing Elo ratings with columns 'From', 'To', and 'Elo'. Each DataFrame should 
      represent the Elo rating history for a specific team.
    
    Returns:
    - float or None: The Elo rating for the specified team on the given date. 
      If no rating is found for that date, the function returns None.
    
    Example:
        elo_rating = find_elo_from_team_df("Team A", pd.to_datetime("2024-09-30"), team_dfs)
    
    Notes:
    - Ensure that the match date is in the same format as the dates in the 'From' and 'To' 
      columns of the DataFrames.
    - The function assumes that the DataFrames for each team are properly formatted and 
      include the necessary columns for filtering.
    """

    # Check if the team has a dataframe
    if team in team_dfs:
        team_df = team_dfs[team]
        # Filter the team_df by the date range
        team_elo = team_df[(team_df['From'] <= match_date) & (team_df['To'] >= match_date)]
        if not team_elo.empty:
            return team_elo['Elo'].values[0]
    return None

def standardize_date_format(date_str):
    """
    Standardizes the format of a date string to 'DD/MM/YYYY'.
    
    This function checks if the provided date string is already in the 'DD/MM/YYYY' format. 
    If it is, the function returns the date string unchanged. If the date string is in a 
    different format (assumed to be 'YYYY-MM-DD'), it converts the date to the 'DD/MM/YYYY' 
    format.
    
    Parameters:
    - date_str (str): A date string that needs to be standardized. It can be in either 
      'DD/MM/YYYY' or 'YYYY-MM-DD' format.
    
    Returns:
    - str: The standardized date string in 'DD/MM/YYYY' format.
    
    Example:
        standardized_date = standardize_date_format("2024-09-30")  # Returns "30/09/2024"
        standardized_date = standardize_date_format("30/09/2024")  # Returns "30/09/2024"
    
    Notes:
    - This function uses regular expressions to determine the current format of the date string.
    - Ensure that the input date string is a valid date; otherwise, the function may raise an 
      error during the conversion process.
    """

    # Check if the date is in 'DD/MM/YYYY' format using regex
    if re.match(r'\d{2}/\d{2}/\d{4}', date_str):
        # If it's already in 'DD/MM/YYYY', no need to change
        return date_str
    # Otherwise, assume it's in 'YYYY-MM-DD' format and convert it to 'DD/MM/YYYY'
    else:
        return pd.to_datetime(date_str).strftime('%d/%m/%Y')

def send_email(df, to_email, today_date):
    """
    Sends an email with a betting summary as an HTML table.
    
    This function converts a given DataFrame into a CSV file and sends it as part of an email 
    body along with a summary of optimal betting propositions.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the betting summary data to be sent in the email.
    - to_email (str): The recipient's email address where the betting summary will be sent.
    - today_date (datetime or str): The date associated with the betting propositions. 
      This is used to name the CSV file and in the email subject line.
    
    Returns:
    - None
    
    Raises:
    - Exception: If the email fails to send, an error message will be printed.
    
    Example:
        send_email(betting_summary_df, "recipient@example.com", "2024-09-30")
    
    Notes:
    - Ensure that the sender's email and password are correctly set.
    - The sender's email account must allow access to less secure apps if using Gmail.
    - The DataFrame is converted to an HTML format for better presentation in the email.
    - This function uses the SMTP protocol to send the email via Gmail's server.
    """

    # Convert the DataFrame to a CSV
    csv_file = f'betting_summary_{str(today_date)}.csv'
    df.to_csv(csv_file, index=False)

    html_content = df.to_html()
    sender_email = "mail@gmail.com"
    sender_password = ""
    subject = f"{str(today_date)} Betting propositions"
    body = f"<h2>Found some optimal bets, enjoy!</h2>{html_content}"
    # Create MIMEMultipart object
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach body
    msg.attach(MIMEText(body, 'html'))

    # Sending the email
    try:
        # Set up the SMTP server connection
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)  # Login

        # Send the email
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()

        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


# In[ ]:




