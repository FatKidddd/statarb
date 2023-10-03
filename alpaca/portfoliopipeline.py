import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pykalman import KalmanFilter
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM

class PortfolioPipeline():
	def __init__(self, p_value_threshold=0.01, max_half_life=60):
	 	# stationarity tests
		self.p_value_threshold = p_value_threshold
		self.min_half_life, self.max_half_life = 0, max_half_life # I want 1 week half life
		self.avg_cross_period_threshold = int(self.max_half_life * 0.75) # i'll just make it less strict for now

	def stationarity_check(self, price_series1, price_series2):
		constant, beta, alpha, residual, adfstat, p_value = self.engle_granger_two_step_cointegration_test(price_series1, price_series2)

		H, half_life, avg_cross_period = None, None, None

		if not(p_value <= self.p_value_threshold):
			return 0, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period # first number is index failed to track failed count

		# Hurst Exponent
		H, c, _data = compute_Hc(residual)
		if H >= 0.5: # spread is not mean-reverting
			return 1, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

		# Half-life - duration to mean-revert
		half_life = -np.log(2) / alpha
		if not(self.min_half_life <= half_life and half_life <= self.max_half_life):
			return 2, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

		# Mean cross frequency
		resid = np.array(residual)
		total_crosses = ((resid[:-1] * resid[1:]) < 0).sum()
		avg_cross_period = len(price_series1) / total_crosses
		if avg_cross_period > self.avg_cross_period_threshold:
			return 3, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period
		
		assert abs(residual.mean()) <= 1e-9

		return -1, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

	def try_validate_pair(self, failed_count, df_data, df, stock1, stock2):
		failed_idx, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period = self.stationarity_check(df[stock1], df[stock2])
		if failed_idx != -1:
			failed_count[failed_idx] += 1
			return

		if beta < 0:
			display(f'Found pair with negative beta - {stock1} {stock2}')
			return

		df_data['Stock1'].append(stock1)
		df_data['Stock2'].append(stock2)
		df_data['Beta'].append(beta)
		df_data['p'].append(p_value)
		df_data['H'].append(H)
		df_data['Half-life'].append(half_life)
		df_data['Avg zero cross period'].append(int(avg_cross_period))

	def log_failed_count(self, failed_count):
		display(f'{failed_count[0]} failed cointegration test')
		display(f'{failed_count[1]} failed H exp criterion')
		display(f'{failed_count[2]} failed half-life criterion')
		display(f'{failed_count[3]} failed avg zero cross period criterion')


	def find_pairs_from_clusters(self, df, clusters):
		cluster_pairs = []
		failed_count = [0]*4
		total_num_of_pairs = 0

		for cluster_idx, cluster in enumerate(clusters):
			n = len(cluster)
			num_of_pairs = n*(n-1)//2
			total_num_of_pairs += num_of_pairs

			cluster_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [] }
			display(f'Testing {num_of_pairs} pairs in cluster {cluster_idx}')

			for i in range(n):
				for j in range(i+1, n):
					self.try_validate_pair(failed_count, cluster_data, df, cluster[i], cluster[j])

			cluster_pairs.append(pd.DataFrame(cluster_data))

		display(f'Tested {total_num_of_pairs} pairs in total')
		self.log_failed_count(failed_count)

		if len(cluster_pairs) == 0:
			return None
		pairs = pd.concat(cluster_pairs, ignore_index=True)
		display(f'Found {pairs.shape[0]} pairs')
		display(pairs)

		return pairs

