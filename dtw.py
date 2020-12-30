def dtw(s, t):
	'''
	Computes and returns the distance between two 
	trajectories (or sets of trajectories).
	'''
	# Change from pandas DataFrame to numpy matrix
	s = s.to_numpy()
	t = t.to_numpy()

	ns = len(s)
	nt = len(t)
	if len(s.columns) != len(t.columns):
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
