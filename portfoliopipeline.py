class PortfolioPipeline(Debugger):
	def __init__(self, p_value_threshold=0.01, max_half_life=60):
		super().__init__()
	 	# stationarity tests
		self.p_value_threshold = p_value_threshold
		self.min_half_life, self.max_half_life = 0, max_half_life # I want 1 week half life
		self.avg_cross_period_threshold = int(self.max_half_life * 0.75) # i'll just make it less strict for now

	def estimate_long_run_short_run_relationships(y, x):
			assert isinstance(y, pd.Series), 'Input series y should be of type pd.Series'
			assert isinstance(x, pd.Series), 'Input series x should be of type pd.Series'
			assert sum(y.isnull()) == 0, 'Input series y has nan-values. Unhandled case.'
			assert sum(x.isnull()) == 0, 'Input series x has nan-values. Unhandled case.'
			assert y.index.equals(x.index), 'The two input series y and x do not have the same index.'
			
			x = sm.add_constant(x)
			long_run_ols = sm.OLS(y, x)
			long_run_ols_fit = long_run_ols.fit()
			
			c, gamma = long_run_ols_fit.params
			z = long_run_ols_fit.resid

			short_run_ols = OLS(y.diff().iloc[1:], (z.shift().iloc[1:]))
			short_run_ols_fit = short_run_ols.fit()
			
			alpha = short_run_ols_fit.params[0]
					
			return c, gamma, alpha, z

	def engle_granger_two_step_cointegration_test(y, x):
			assert isinstance(y, pd.Series), 'Input series y should be of type pd.Series'
			assert isinstance(x, pd.Series), 'Input series x should be of type pd.Series'
			assert sum(y.isnull()) == 0, 'Input series y has nan-values. Unhandled case.'
			assert sum(x.isnull()) == 0, 'Input series x has nan-values. Unhandled case.'
			assert y.index.equals(x.index), 'The two input series y and x do not have the same index.'
			
			c, gamma, alpha, z = estimate_long_run_short_run_relationships(y, x)
			
			# NOTE: The p-value returned by the adfuller function assumes we do not estimate z first, but test 
			# stationarity of an unestimated series directly. This assumption should have limited effect for high N, 
			# so for the purposes of this course this p-value can be used for the EG-test. Critical values taking 
			# this into account more accurately are provided in e.g. McKinnon (1990) and Engle & Yoo (1987).
			
			adfstat, pvalue, usedlag, nobs, crit_values = adfuller(z, maxlag=1, autolag=None)
		
			return c, gamma, alpha, z, adfstat, pvalue

	def stationarity_check(self, df, stock1, stock2):
		price_series1, price_series2 = df[stock1], df[stock2]

		constant, beta, alpha, residual, adfstat, p_value = engle_granger_two_step_cointegration_test(price_series1, price_series2)

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

	def find_pairs_from_clusters(self, df, clusters):
		cluster_pairs = []
		failed_count = [0]*4
		total_num_of_pairs = 0

		for cluster_idx, cluster in enumerate(clusters):
			n = len(cluster)
			num_of_pairs = n*(n-1)//2
			total_num_of_pairs += num_of_pairs

			cluster_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [] }
			self._log(f'Testing {num_of_pairs} pairs in cluster {cluster_idx}')

			for i in range(n):
				for j in range(i+1, n):
					failed_idx, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period = self.stationarity_check(df, cluster[i], cluster[j])
					if failed_idx != -1:
						failed_count[failed_idx] += 1
						continue

					if beta < 0:
						self._log(f'Found pair with negative beta - {cluster[i]} {cluster[j]}')
						continue

					cluster_data['Stock1'].append(cluster[i])
					cluster_data['Stock2'].append(cluster[j])
					cluster_data['Beta'].append(beta)
					cluster_data['p'].append(p_value)
					cluster_data['H'].append(H)
					cluster_data['Half-life'].append(half_life)
					cluster_data['Avg zero cross period'].append(int(avg_cross_period))
					cluster_data['Cluster'].append(int(cluster_idx))


			cluster_pairs.append(pd.DataFrame(cluster_data))

		self._log(f'Tested {total_num_of_pairs} pairs in total')
		self._log(f'{failed_count[0]} failed cointegration test')
		self._log(f'{failed_count[1]} failed H exp criterion')
		self._log(f'{failed_count[2]} failed half-life criterion')
		self._log(f'{failed_count[3]} failed avg zero cross period criterion')

		if len(cluster_pairs) == 0:
			return None
		pairs = pd.concat(cluster_pairs, ignore_index=True)
		self._log(f'Found {pairs.shape[0]} pairs')
		self._log(pairs)

		return pairs

	def filter_positive_pnl_pairs(self, validated_pairs, sorted_validation_backtest_results, min_validation_return_threshold=0.1):
		positive_pnl_pairs = list(filter(lambda x: x[-1][-1]-x[-1][0]>0, sorted_validation_backtest_results))

		filtered_validated_pairs_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [], 'Best entry z threshold': [], 'Best exit z threshold': [], 'Validation Fees': [], 'Validation PnL': [] }

		for (pair_df, training_pair_df, stock1, stock2, beta, best_entry_z_threshold, best_exit_z_threshold, best_position, best_fees, best_margin) in positive_pnl_pairs:
			validation_pnl = best_margin[-1]-best_margin[0]
			validation_return = (best_margin[-1]-best_margin[0]) / best_margin[0]
			if validation_return < min_validation_return_threshold:
				continue
			df = validated_pairs.loc[(validated_pairs['Stock1']==stock1) & (validated_pairs['Stock2']==stock2), :]
			for col in df.columns:
				filtered_validated_pairs_data[col].append(df[col].values[0])
			filtered_validated_pairs_data['Best entry z threshold'].append(best_entry_z_threshold)
			filtered_validated_pairs_data['Best exit z threshold'].append(best_exit_z_threshold)
			filtered_validated_pairs_data['Validation Fees'].append(np.sum(best_fees))
			filtered_validated_pairs_data['Validation PnL'].append(validation_pnl)

		filtered_validated_pairs = pd.DataFrame(filtered_validated_pairs_data)

		self._log(f'Sector {sector}: {len(positive_pnl_pairs)}/{len(sorted_validation_backtest_results)} = {len(positive_pnl_pairs) / len(sorted_validation_backtest_results)*100:.2f}% have +PnL')
		self._log(filtered_validated_pairs)
		return filtered_validated_pairs


	def revalidate_pairs(self, df, validated_pairs, sector):
		test_pairs_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [], 'Best entry z threshold': [], 'Best exit z threshold': [], 'Validation Fees': [], 'Validation PnL': [], 'Sector': [] }
		failed_count = [0]*4
		old_num_pairs = validated_pairs.shape[0]

		self._log(f'Testing {old_num_pairs} pairs')

		for stock1, stock2, _, _, _, _, _, cluster, best_entry_z_threshold, best_exit_z_threshold, validation_fees, validation_pnl in validated_pairs.values:
			failed_idx, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period = self.stationarity_check(df, stock1, stock2)
			if failed_idx != -1:
				failed_count[failed_idx] += 1
				continue

			test_pairs_data['Stock1'].append(stock1)
			test_pairs_data['Stock2'].append(stock2)
			test_pairs_data['Beta'].append(beta)
			test_pairs_data['p'].append(p_value)
			test_pairs_data['H'].append(H)
			test_pairs_data['Half-life'].append(half_life)
			test_pairs_data['Avg zero cross period'].append(int(avg_cross_period))
			test_pairs_data['Cluster'].append(int(cluster))
			test_pairs_data['Best entry z threshold'].append(best_entry_z_threshold)
			test_pairs_data['Best exit z threshold'].append(best_exit_z_threshold)
			test_pairs_data['Validation Fees'].append(validation_fees)
			test_pairs_data['Validation PnL'].append(validation_pnl)
			test_pairs_data['Sector'].append(sector)

		test_pairs = pd.DataFrame(test_pairs_data)
		new_num_pairs = test_pairs.shape[0]

		self._log(f'{new_num_pairs}/{old_num_pairs} ({new_num_pairs/old_num_pairs*100:.2f}%) passed stationary check')
		self._log(f'{failed_count[0]} failed cointegration test')
		self._log(f'{failed_count[1]} failed H exp criterion')
		self._log(f'{failed_count[2]} failed half-life criterion')
		self._log(f'{failed_count[3]} failed avg zero cross period criterion')

		self._log(test_pairs)

		return test_pairs
