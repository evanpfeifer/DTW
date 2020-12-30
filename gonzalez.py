import pandas as pd
import numpy as np
import random
# Assumes that the dynamic time warp function from dtw.py is already imported


def gonzalez_cluster(positions, k_clusters=10, offence=True, starting_cluster_ID, seed=100):
	'''
	positions: Pandas DataFrame where each row is one frame for one player
	k_clusters: Number of clusters to create.
	offence: Boolean, whether you want the offence or defence data.
	starting_cluster_ID: What number to start cluster IDs at.
	seed: Integer, seed for the random number generator.
	'''
	cluster_ID_counter = starting_cluster_ID
	num_clusters = 0
	random.seed(100)
	info_table_index = positions.index.unique()

	# Make initial cluster
	curr_center = info_table_index[random.randint(0, len(info_table_index))]
	info_table = pd.DataFrame(data=[[curr_center, 0, cluster_ID_counter]], 
		columns=["Cluster", "DistToCenter", "ClusterID"], 
		index=[cluster_ID_counter])
	curr_center_traj = get_traj(curr_center, offence)
	for i in info_table_index:
		if i == curr_center:
			continue
		compare_traj = get_traj(i, offence)
		info_table.loc[i] = [curr_center, dtw(curr_center_traj, compare_traj), cluster_ID_counter]
	info_table.sort_values(by=['DistToCenter'], axis=1, ascending=False)
	cluster_ID_counter += 1
	num_clusters += 1

	# Keep adding clusters until finished
	while num_clusters < k_clusters:
		curr_center = info_table.index[0]
		curr_center_traj = get_traj(curr_center, offence)
		for i in info_table_index:
			if i == curr_center:
				info_table.loc[i] = [curr_center, 0, cluster_ID_counter]
				continue
			compare_traj = get_traj(i, offence)
			new_dist = dtw(curr_center_traj, compare_traj)
			if new_dist < info_table.loc[i, "DistToCenter"]:
				info_table.loc[i] = [curr_center, new_dist, cluster_ID_counter]
		info_table.sort_values(by=['DistToCenter'], axis=1, ascending=False)
		cluster_ID_counter += 1
		num_clusters += 1

	# Return the information table with mappings
	return info_table
