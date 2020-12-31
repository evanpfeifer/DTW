# Imports

import numpy as np
import pandas as pd
import math

week_1 = pd.read_csv('week1.csv')
week_2 = pd.read_csv('week2.csv')
week_3 = pd.read_csv('week3.csv')
week_4 = pd.read_csv('week4.csv')
week_5 = pd.read_csv('week5.csv')
week_6 = pd.read_csv('week6.csv')
week_7 = pd.read_csv('week7.csv')
week_8 = pd.read_csv('week8.csv')
week_9 = pd.read_csv('week9.csv')
week_10 = pd.read_csv('week10.csv')
week_11 = pd.read_csv('week11.csv')
week_12 = pd.read_csv('week12.csv')
week_13 = pd.read_csv('week13.csv')
week_14 = pd.read_csv('week14.csv')
week_15 = pd.read_csv('week15.csv')
week_16 = pd.read_csv('week16.csv')
week_17 = pd.read_csv('week17.csv')

all_weeks = [week_1, week_2, week_3, week_4, week_5, week_6,
            week_7, week_8, week_9, week_10, week_11, week_12, 
            week_13, week_14, week_15, week_16, week_17]




games = pd.read_csv('games.csv')
players = pd.read_csv('players.csv')
plays = pd.read_csv('plays.csv')
# Add a unique Play ID to the plays data for easier use
plays['uniquePlayId'] = plays['gameId'] * 10000 + plays['playId']




positions = pd.concat(all_weeks)
# Give the football an nflId of 9999999
positions['nflId'].fillna(float(9999999), inplace=True)

# Add a unique Play ID to the positional data for easier use
positions['uniquePlayId'] = positions['gameId'] * 10000 + positions['playId']

# Give uniue ID for each combination of player and play
def make_identifier(df):
    str_id = df.apply(lambda x: '_'.join(map(str, x)), axis=1)
    return pd.factorize(str_id)[0]

positions['playAndPlayerId'] = make_identifier(positions[['uniquePlayId','nflId']])




# Normalize positions for play direction to always be to the right

# Split to find plays where the direction is 'left'
# This way we can utilize array vectorization (pd.DataFrame.apply too slow)
positionsGoLeft = positions[positions['playDirection'] == 'left']
positionsGoRight = positions[positions['playDirection'] == 'right']

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Apply linear transformation
positionsGoLeft['x'] = 120 - positionsGoLeft['x']
positionsGoLeft['y'] = 53.3 - positionsGoLeft['y']

# Concatenate positions table back together
positions = pd.concat([positionsGoLeft, positionsGoRight])
positions = positions.drop('playDirection', axis=1)




# Join table to add whether a player is on offence/defence on a given play
plays_with_teams = plays.merge(games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']], on ='gameId')
positions = positions.merge(plays_with_teams[['uniquePlayId', 'possessionTeam', 
                                  'homeTeamAbbr', 'visitorTeamAbbr']], on='uniquePlayId')

positions['homePossession'] = positions['possessionTeam'] == positions['homeTeamAbbr']
positions.loc[(positions['team'] == 'away') & (positions['homePossession'] == False), 'team'] = 'offence'
positions.loc[(positions['team'] == 'away') & (positions['homePossession'] == True), 'team'] = 'defence'
positions.loc[(positions['team'] == 'home') & (positions['homePossession'] == True), 'team'] = 'offence'
positions.loc[(positions['team'] == 'home') & (positions['homePossession'] == False), 'team'] = 'defence'




# Make new column for altered positions
positions['newPos'] = ''

posCopy = positions.copy()
posCopy['indices'] = posCopy.index
playToIndex = posCopy[['uniquePlayId', 'indices']].groupby('uniquePlayId', as_index=False).agg(list)
playToIndex.set_index('uniquePlayId', inplace=True)

listOfSeries = []

for uniquePlayId in playToIndex.index:
    # Each play has offence and defence
    for team_type in ['offence', 'defence']:
        onePlayFrame = positions.iloc[playToIndex.loc[uniquePlayId, 'indices'], :]
        onePlayFrame = onePlayFrame[(onePlayFrame['frameId'] == 1) 
                            & (onePlayFrame['frameId'] == 1)] 
                            # & (onePlayFrame['team'] == team_type)]
        if team_type == 'offence':
            onePlayFrame = onePlayFrame[(onePlayFrame['team'] == team_type) | (onePlayFrame['team'] == 'football')]
        else:
            onePlayFrame = onePlayFrame[onePlayFrame['team'] == team_type]
        onePlayFrame = onePlayFrame[['time', 'x', 'y', 'nflId', 'position', 
                        'team', 'uniquePlayId', 'frameId', 'playAndPlayerId']].sort_values(['position', 'y'], ascending=False)
        onePlayFrame.set_index('playAndPlayerId', inplace=True)

        a = onePlayFrame['position'].copy()
        b = onePlayFrame['position'].copy()

        placeholder = 2
        for i in np.arange(1, len(a)):
            if a.iloc[i] == a.iloc[i - 1]:
                b.iloc[i] = b.iloc[i] + str(placeholder)
                placeholder += 1
            else:
                placeholder = 2
        
        listOfSeries.append(b)




new_pos_series = pd.concat(listOfSeries)
# nan values are the football
new_pos_series.fillna('FTBL', inplace=True)
new_pos_series = new_pos_series.sort_index()




for i in np.arange(len(positions)):
    positions.at[i, 'newPos'] = new_pos_series[positions.at[i, 'playAndPlayerId']]




positions.to_csv('cleanData.csv')


















