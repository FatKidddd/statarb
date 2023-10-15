import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from utils import display_side_by_side, ffill_and_dropna

class ClusterPipeline():
	def __init__(self, pca_factors=15, min_samples=3, xi=0.05):
		self.pca_factors = pca_factors
		self.min_samples = min_samples
		self.xi = xi

	def find_clusters(self, df):
		R = df.pct_change().iloc[1:, :] # here the columns of R are the different observations.
		R, dropped = ffill_and_dropna(R, 7) # avoid any stocks with missing returns
		norm_R = (R - R.mean()) / R.std()
		norm_R, dropped = ffill_and_dropna(norm_R, 7) # avoid any stocks with missing returns

		pca = PCA()
		pca.fit(norm_R.T) # use returns as columns and stocks as rows
		pca_data = pca.transform(norm_R.T) # get PCA coordinates for scaled_data

		X = pca_data[:, :self.pca_factors]
		X = pd.DataFrame(X, columns=['PC'+str(i) for i in range(1, self.pca_factors+1)], index=norm_R.columns)

		display(f'{np.sum(pca.explained_variance_ratio_[:self.pca_factors] * 100)}% of variance - {self.pca_factors} components')

		optics_model = OPTICS(min_samples=self.min_samples, xi=self.xi)
		# min_samples parameter -> min number of samples required to form a dense region
		# xi parameter -> max distance between two samples to be considered as a neighborhood
		# min_cluster_size -> min size of a dense region to be considered as a cluster

		clustering = optics_model.fit(X)
		clusters = []
		for i in range(max(1, len(set(optics_model.labels_))-1)):
			cluster = list(X[optics_model.labels_==i].index)
			clusters.append(cluster)

		# PLOT

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

		colors = ['r', 'g','b','c','y','m', 'coral', 'darkgreen', 'crimson', 'darkblue', 'ivory', 'khaki'] * 5

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

		display_side_by_side([pd.DataFrame({'Name': cluster}) for cluster in clusters], 'Cluster ')

		return clusters



