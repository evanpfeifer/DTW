import numpy as np
import pandas as pd
import random
pd.options.mode.chained_assignment = None
import math




positions = pd.read_csv('cleanData.csv')

playPosOffGroup = positions[(positions['team'] == 'offence') 
                            | (positions['team'] == 'football')][['uniquePlayId', 'newPos']].groupby('uniquePlayId').agg(set)
playPosDefGroup = positions[positions['team'] == 'defence'][['uniquePlayId', 'newPos']].groupby('uniquePlayId').agg(set)

positions = positions.set_index('uniquePlayId')

# Print out status
print('Data successfully loaded')

# Set initial player group
curr_players_set = {'FTBL', 'QB', 'RB', 'TE', 'TE2', 'WR', 'WR2'}
curr_players_list = list(curr_players_set)

temp = playPosOffGroup[playPosOffGroup['newPos'] == curr_players_set]

testPositions = positions.loc[temp.index, :]

def get_traj(uniquePlayId, offence):
    '''
    Return a DataFrame for trajectories of either the offence or
    defence. The 'offence' parameter should be a boolean.
    '''
    if offence:
        team = "offence"
    else:
        team = "defence"
    
    play = testPositions.loc[uniquePlayId, ('x', 'y', 'frameId', 'team', 'newPos')]
    play = play[(play['team'] == team) | (play['team'] == 'football')]
    play = play.drop(columns=['team'])
    play = play.set_index(['frameId', 'newPos']).stack().unstack([1, 2])

    footballX = play['FTBL']['x'].iloc[0]
    footballY = play['FTBL']['y'].iloc[0]
    
    # Only keep football positional data for offensive trajectories
    if team != 'offence':
        play = play.drop('FTBL', axis=1)
        
    # Normlize all positions to the starting location of the football
    for col in play.columns.get_level_values(0).unique():
        play[col]['x'] -= footballX
        play[col]['y'] -= footballY
    
    # Reorder columns
    play = play[curr_players_list]
    
    return play

def dtw(s, t):
    # Change from pandas DataFrame to numpy matrix
    s = s.to_numpy()
    t = t.to_numpy()
    
    if len(s.shape) == 1 or len(t.shape) == 1:
        ns = s.shape[0]
        nt = t.shape[0]
        # initialization
        D = np.matrix(np.ones((ns+1, nt+1)) * np.inf)
        D[0, 0] = 0
        # begin dynamic programming
        for i in range(ns):
            for j in range(nt):
                oost = np.linalg.norm(s[i] - t[j])
                D[i + 1, j + 1] = oost + min(D[i, j + 1], D[i + 1, j], D[i, j])

        return D[ns, nt]
    
    else:
        ns, s_cols = s.shape
        nt, t_cols = t.shape
        
        if s_cols != t_cols:
            raise Exception("Mismatching columns")
            
        # initialization
        D = np.matrix(np.ones((ns+1, nt+1)) * np.inf)
        D[0, 0] = 0
        # begin dynamic programming
        for i in range(ns):
            for j in range(nt):
                oost = np.linalg.norm(s[i, :] - t[j, :])
                D[i + 1, j + 1] = oost + min(D[i, j + 1], D[i + 1, j], D[i, j])

        return D[ns, nt]

def gonzalez_cluster(positions, k_clusters, offence, starting_cluster_ID, seed):
    '''
    positions: Pandas DataFrame where each row is one frame for one player
    k_clusters: Number of clusters to create.
    offence: Boolean, whether you want the offence or defence data.
    starting_cluster_ID: What number to start cluster IDs at.
    seed: Integer, seed for the random number generator.
    '''
    cluster_ID_counter = starting_cluster_ID
    num_clusters = 0
    random.seed(seed)
    info_table_index = positions.index.unique()


    playID_to_traj = dict()
    for i in info_table_index:
        playID_to_traj[i] = get_traj(i, offence)

    curr_center = info_table_index[random.randint(0, len(info_table_index))]
    info_table = pd.DataFrame(data=[[curr_center, 0, cluster_ID_counter]], 
        columns=["Cluster", "DistToCenter", "ClusterID"], 
        index=[curr_center])
    curr_center_traj = playID_to_traj[curr_center]
    for i in info_table_index:
        if i == curr_center:
            continue
        compare_traj = playID_to_traj[i]
        info_table.loc[i] = [curr_center, dtw(curr_center_traj, compare_traj), cluster_ID_counter]
    info_table.sort_values(by=["DistToCenter"], ascending=False, inplace=True)
    cluster_ID_counter += 1
    num_clusters += 1

    # Print where we are at
    print("Currently at k = ", num_clusters, " clusters")

    # Keep adding clusters until finished
    while num_clusters < k_clusters:
        curr_center = info_table.index[0]
        curr_center_traj = playID_to_traj[curr_center]
        for i in info_table_index:
            if i == curr_center:
                info_table.loc[i] = [curr_center, 0, cluster_ID_counter]
                continue
            compare_traj = playID_to_traj[i]
            new_dist = dtw(curr_center_traj, compare_traj)
            if new_dist < info_table.loc[i, "DistToCenter"]:
                info_table.loc[i] = [curr_center, new_dist, cluster_ID_counter]
        info_table.sort_values(by=["DistToCenter"], ascending=False, inplace=True)
        cluster_ID_counter += 1
        num_clusters += 1
        # Print out where we are at
        print("Currently at k = ", num_clusters, " clusters")

    # Return the information table with mappings
    return info_table

testClusters = gonzalez_cluster(testPositions, 35, True, 0, 100)

clusterGroupings = testClusters.groupby('ClusterID').count()

testClusters.to_csv('boi.csv')
clusterGroupings.to_csv('boiGroups.csv')







