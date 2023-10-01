class ClusterPipeline(Debugger):
	def __init__(self, df, pca_factors=15, min_samples=3):
		super().__init__()
		self.pca_factors = pca_factors
		self.min_samples = min_samples

		return self.find_clusters(df)

	def _ffill_and_dropna(self, df, caption, limit=None):
		old_shape = df.shape[1]
		df.fillna(method='ffill', inplace=True, limit=limit)
		df.dropna(axis='columns', inplace=True)
		self._log(f'{caption} - {old_shape} to {df.shape[1]} columns - {old_shape - df.shape[1]} NA columns dropped')

	def find_clusters(self, df):
		R = df.pct_change().iloc[1:, :] # here the columns of R are the different observations.
		self._ffill_and_dropna(R, 'R', 10) # avoid any stocks with missing returns
		norm_R = (R - R.mean()) / R.std()
		self._ffill_and_dropna(norm_R, 'Norm R', 10) # avoid any stocks with missing returns

		pca = PCA()
		pca.fit(norm_R.T) # use returns as columns and stocks as rows
		pca_data = pca.transform(norm_R.T) # get PCA coordinates for scaled_data

		X = pca_data[:, :self.pca_factors]
		X = pd.DataFrame(X, columns=['PC'+str(i) for i in range(1, self.pca_factors+1)], index=norm_R.columns)

		self._log(f'{np.sum(pca.explained_variance_ratio_[:self.pca_factors] * 100)}% of variance - {self.pca_factors} components')

		optics_model = OPTICS(min_samples=self.min_samples)
		# min_samples parameter -> min number of samples required to form a dense region
		# xi parameter -> max distance between two samples to be considered as a neighborhood
		# min_cluster_size -> min size of a dense region to be considered as a cluster

		clustering = optics_model.fit(X)
		clusters = []
		for i in range(len(set(optics_model.labels_))-1):
			cluster = list(X[optics_model.labels_==i].index)
			clusters.append(cluster)

		self.plot_clusters(pca, optics_model, X)

		return clusters

	def plot_clusters(self, pca, optics_model, X):
		if not self.display_graphs:
			return
		# PCA plot
		per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
		labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
		plt.figure(figsize=(5, 3))
		plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
		plt.ylabel('Percentage of Explained Variance')
		plt.xlabel('Principal Component')
		plt.title('Scree Plot')
		plt.show()

		# Cluster plots
		space = np.arange(len(X))
		reachability = optics_model.reachability_[optics_model.ordering_]
		labels = optics_model.labels_[optics_model.ordering_]

		plt.figure(figsize=(12, 3))
		G = gridspec.GridSpec(1, 3)
		ax1 = plt.subplot(G[0, :2])
		ax2 = plt.subplot(G[0, -1])

		colors = ['r', 'g','b','c','y','m', 'coral', 'darkgreen', 'crimson', 'darkblue', 'ivory', 'khaki', 'r', 'g','b','c','y','m', 'coral', 'darkgreen', 'crimson', 'darkblue', 'ivory', 'khaki', 'r', 'g','b','c','y','m']

		assert len(set(labels)) <= len(colors)

		for i, color in enumerate(colors):
			Xk = space[labels == i]
			Rk = reachability[labels == i]
			ax1.plot(Xk, Rk, color, alpha = 0.3, marker='.')
			ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha = 0.3)
			ax1.plot(space, np.full_like(space, 2., dtype = float), 'k-', alpha = 0.5)
			ax1.plot(space, np.full_like(space, 0.5, dtype = float), 'k-.', alpha = 0.5)
			ax1.set_ylabel('Reachability Distance')
			ax1.set_title('Reachability Plot')

		# Plotting the OPTICS Clustering
		for i, color in enumerate(colors):
			Xk = X[optics_model.labels_ == i]
			ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], color, alpha = 0.3, marker='.')
			ax2.plot(X.iloc[optics_model.labels_ == -1, 0], X.iloc[optics_model.labels_ == -1, 1],'k+', alpha = 0.1)
			ax2.set_title('OPTICS Clustering')

		plt.tight_layout()
		plt.show()

		self.display_side_by_side([pd.DataFrame({'Name': map(self.symbol_helper, cluster)}) for cluster in clusters], 'Cluster ')
