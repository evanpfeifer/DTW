# Gonzalez Clustering, Dynamic Time Warp, and trajectory simplification

import numpy as np
import pandas as pd
from scipy.spatial import distance
import random
pd.options.mode.chained_assignment = None
import math




positions = pd.read_csv('cleanData.csv')

playPosOffGroup = positions[(positions['team'] == 'offense') 
                            | (positions['team'] == 'football')][['uniquePlayId', 'newPos']].groupby('uniquePlayId').agg(set)
playPosDefGroup = positions[positions['team'] == 'defense'][['uniquePlayId', 'newPos']].groupby('uniquePlayId').agg(set)

positions = positions.set_index('uniquePlayId')

# Print out status
print('Data successfully loaded')



def linearSimp(P, epsilonDist):
    '''
    Simplifies a trajectory - reduces the number of vertices based on an
    epsilon value. Based on the Driemel et al. (2012) method.

    P = formatted pandas df with data for one play (i.e. what get_traj returns)
    '''
    pos1 = 0
    pos2 = 1
    pos1to2dist = 0
    # Get current trajectory size
    sP = len(P)

    if sP <= 2:
        return P
    # Initialize the new traj with the first vertex
    newTraj = P.iloc[[pos1]]
    # Index counter for num rows in newTraj
    newTrajIndex = 1

    for i in range(pos2, sP):
        pos1to2dist = distance.euclidean(P.iloc[pos1], P.iloc[pos2])
        if i == sP - 1: # At last vertex
            doneTraj = True
            # Only add last vertex if it is diff than prev vertex
            if pos1to2dist != 0:
                newTraj.loc[newTrajIndex] = P.iloc[i] 
                newTrajIndex += 1
        elif pos1to2dist > epsilonDist:
            newTraj.loc[newTrajIndex] = P.iloc[i]
            pos1 = i
            pos2 = pos1 + 1
            newTrajIndex += 1
        else: # We can eliminate this vertex in the simplification
            pos2 += 1

    if len(newTraj) == 1: # Simplified traj is only a single vertex
        newTraj = newTraj.append(newTraj) # So add one more vertex
        newTraj.index = [1, 2]

    return newTraj

def get_traj(uniquePlayId, offense):
    '''
    Return a DataFrame for trajectories of either the offense or
    defense. The 'offense' parameter should be a boolean.
    '''
    if offense:
        team = "offense"
    else:
        team = "defense"
    
    play = testPositions.loc[uniquePlayId, ('x', 'y', 'frameId', 'team', 'newPos')]
    play = play[(play['team'] == team) | (play['team'] == 'football')]
    play = play.drop(columns=['team'])
    play = play.set_index(['frameId', 'newPos']).stack().unstack([1, 2])

    footballX = play['FTBL']['x'].iloc[0]
    footballY = play['FTBL']['y'].iloc[0]
    
    # Only keep football positional data for offensive trajectories
    if team != 'offense':
        play = play.drop('FTBL', axis=1)
        
    # Normlize all positions to the starting location of the football
    for col in play.columns.get_level_values(0).unique():
        play[col]['x'] -= footballX
        play[col]['y'] -= footballY
    
    # Reorder columns
    play = play[curr_players_list]

    # Check for nans
    for row in range(len(play)):
        if play.iloc[row].isnull().values.any():
            play = play.iloc[0:row]
            break
    
    epsilonDist = 3 # Simplify trajectories with 3 yard distances
    return linearSimp(play, epsilonDist)

def dtw(s, t):
    '''
    Computes the Dynamic Time Warp distance between two sets
    of trajectories
    '''
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

def gonzalez_cluster(positions, max_radius, offense, starting_cluster_ID, seed):
    '''
    positions: Pandas DataFrame where each row is one frame for one player
    max_radius: Creates clusters until the cluster with the largest radius has a
                radius less than max_radius
    offense: Boolean, whether you want the offense or defense data.
    starting_cluster_ID: What number to start cluster IDs at.
    seed: Integer, seed for the random number generator.
    '''
    cluster_ID_counter = starting_cluster_ID
    num_clusters = 0
    random.seed(seed)
    info_table_index = positions.index.unique()


    playID_to_traj = dict()
    for i in info_table_index:
        playID_to_traj[i] = get_traj(i, offense)

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
    print("___________________________________________________")
    print("Currently at k = ", num_clusters, " clusters")

    # Keep adding clusters until finished
    biggest_radius = info_table.iloc[0, 1]
    while biggest_radius > max_radius:
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
        print("___________________________________________________")
        print("Currently at k = ", num_clusters, " clusters")
        biggest_radius = info_table.iloc[0, 1]
        if num_clusters % 5 == 0:
            print("Current largest cluster radius: ", biggest_radius)

    # Return the information table with mappings
    return info_table, num_clusters



start_index = 0
my_counter = 1
return_table = None




# ##################################
#
# Choose appropriate settings for
# offensive or defensive players
#
# ##################################

# player_groups = playPosDefGroup['newPos'].apply(sorted).apply(tuple).value_counts()[:326].index
# player_groups = playPosDefGroup['newPos'].apply(sorted).apply(tuple).value_counts()[:326].index

# player_sets_list = []
# for i in player_groups:
#     player_sets_list += [set(i)]

# for player_set in player_sets_list:
#     # Set initial player group
#     curr_players_set = player_set
#     curr_players_list = list(curr_players_set)
#     temp = playPosOffGroup[playPosOffGroup['newPos'] == curr_players_set]
#     testPositions = positions.loc[temp.index, :]

#     testClusters, num_k = gonzalez_cluster(testPositions, 1000, True, start_index, 100)
#     print("###################################################")
#     print("Total clusters = ", num_k)
#     print("Finished ", my_counter, " of 326 player sets")
#     print("###################################################")

#     if my_counter == 1:
#         return_table = testClusters
#     else:
#         return_table = return_table.append(testClusters)
#     start_index += num_k
#     my_counter += 1

# return_table.to_csv("defense_cluster.csv")
