class DataPipeline(Debugger):
	def __init__(self, tickers, start_date, validation_start_date, testing_start_date, end_date):
		super().__init__()

		self.tickers = tickers
		for ticker in self.tickers:
			symbol = self.qb.AddEquity(ticker, Resolution.Hour).Symbol # same as using ticker itself

		self.start_date = start_date
		self.end_date = end_date
		self.validation_start_date = validation_start_date
		self.testing_start_date = testing_start_date

		self.raw_history = self.qb.History(self.qb.Securities.Keys, self.start_date, self.end_date, Resolution.Hour)
		self.df = self.raw_history['close'].unstack(level=0)

		# coordinate start timings for training data
		self.coordinate_start_timings(percent=0.95) 

		# remove outlier data points
		self.remove_outlier_data(max_return=0.2)

		# filter by volume
		# self.filter_selection(min_volume=, min_price_=, )

		self.training_df = self.df.loc[self.df.index < self.validation_start_date]
		self.validation_df = self.df.loc[(self.df.index >= self.validation_start_date) & (self.df.index < self.testing_start_date)]
		self.testing_df = self.df.loc[self.df.index >= self.testing_start_date]
		self.training_and_validation_df = self.df.loc[self.df.index < self.testing_start_date]

		self._log(f'Dataset {self.df.shape[0]} = {self.training_df.shape[0]} + {self.validation_df.shape[0]} + {self.testing_df.shape[0]}') 
		self._log(f'Train + validation = {self.training_and_validation_df.shape[0]}') 

		return self.training_df, self.validation_df, self.testing_df, self.training_and_validation_df

	def coordinate_start_timings(self, percent=0.9):
		# need to coordinate start timings for etfs because some were created later
		old_num_cols = self.df.shape[1]
		old_num_rows = self.df.shape[0]
		self.df.fillna(method='ffill', inplace=True, limit=50) # removed discontinued etfs whose data is ffill
		new_df = self.df.dropna(axis='columns', thresh=int(percent*old_num_rows))
		# self._log(self.df[np.setdiff1d(self.df.columns, new_df.columns, assume_unique=True)])
		self.df = new_df
		self._log(f'{old_num_cols} to {self.df.shape[1]} columns - {old_num_cols - self.df.shape[1]} NA columns dropped')
		idx_to_start = self.df.notnull().all(axis=1).argmax() # first common non na value

		self._log(f'Actual start date {self.df.index[idx_to_start]}, removed first {idx_to_start} or {idx_to_start/old_num_rows*100:.2f}% rows')
		self.df = self.df.iloc[idx_to_start:]

	def remove_outlier_data(self, max_return_threshold=0.2):
		outlier_returns = self.df.pct_change().abs() > max_return_threshold
		summed_outlier_returns = outlier_returns.sum()
		self._log(summed_outlier_returns[summed_outlier_returns > 0])
		self.df[outlier_returns] = np.nan # remove outlier data
		self._ffill_and_dropna(self.df, 50) # removed discontinued etfs whose data is ffill

	def consec_repeat_starts(a, n):
		N = n-1
		m = a[:-1]==a[1:]
		return np.flatnonzero(np.convolve(m,np.ones(N, dtype=int))==N)-N+1	

	def filter_selection(self):
		pass

	def _ffill_and_dropna(self, df, limit=None):
		old_shape = df.shape[1]
		df.fillna(method='ffill', inplace=True, limit=limit)
		df.dropna(axis='columns', inplace=True)
		self._log(f'{old_shape} to {df.shape[1]} columns - {old_shape - df.shape[1]} NA columns dropped')

	def plot_data(self):
		pass