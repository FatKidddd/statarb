from AlgorithmImports import *
from hurst import compute_Hc
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

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

class PairsPCAOPTICS(QCAlgorithm):
	def Initialize(self):
		self.initial_capital = 30000
		self.SetCash(self.initial_capital)
		self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

		self.SetStartDate(2018, 1, 1)

		self.pair_tickers = [['EFO UD63CSAA26P1', 'UPV UM61FJMT8EHX']]
		for ticker in np.unique(self.pair_tickers).tolist():
			symbol = self.AddEquity(ticker, Resolution.Hour).Symbol

		self.formation_period = 252 * 4 * 6 
		self.lookback_period = 252 * 6

		self.raw_history = self.History(self.Securities.Keys, self.formation_period, Resolution.Hour)
		self.df = self.raw_history['close'].unstack(level=0)
		self.df = self.df.iloc[:-1, :] # history will give overlap for first data point so we remove it

		self.coordinate_start_timings(percent=0.95) # coordinate start timings for training data
		self.remove_outlier_data(max_return_threshold=0.2) # remove outlier data points

		self.validation_df = self.df.iloc[-self.lookback_period:, :]

		self.Log(f'{self.df.shape[0]}, {self.validation_df.shape[0]}')

	 	# stationarity test params
		self.p_value_threshold = 0.01
		self.min_half_life, self.max_half_life = 0, 10 * 6 # I want about 1 week half life
		self.avg_cross_period_threshold = int(self.max_half_life * 0.75) # i'll just make it less strict for now

		# get pairs data
		self.pairs = self.get_pairs_data(self.pair_tickers)
		assert self.pairs.shape[0] == len(self.pair_tickers)
		self.pairs_mean, self.pairs_std, self.pairs_z_threshold = self.get_best_z_thresholds(self.pairs)

		self.positions = { pair_key: (0, 0) for pair_key in self.pairs_mean }

	def coordinate_start_timings(self, percent=0.9):
		# need to coordinate start timings for etfs because some were created later
		old_num_cols = self.df.shape[1]
		old_num_rows = self.df.shape[0]
		self.df.fillna(method='ffill', inplace=True, limit=50) # removed discontinued etfs whose data is ffill
		new_df = self.df.dropna(axis='columns', thresh=int(percent*old_num_rows))
		# self.Log(self.df[np.setdiff1d(self.df.columns, new_df.columns, assume_unique=True)])
		self.df = new_df
		self.Log(f'{old_num_cols} to {self.df.shape[1]} columns - {old_num_cols - self.df.shape[1]} NA columns dropped')
		idx_to_start = self.df.notnull().all(axis=1).argmax() # first common non na value

		self.Log(f'Actual start date {self.df.index[idx_to_start]}, removed first {idx_to_start} or {idx_to_start/old_num_rows*100:.2f}% rows')
		self.df = self.df.iloc[idx_to_start:]

	def remove_outlier_data(self, max_return_threshold=0.2):
		outlier_returns = self.df.pct_change() > max_return_threshold
		summed_outlier_returns = outlier_returns.sum()
		self.Log(summed_outlier_returns[summed_outlier_returns > 0])
		self.df[outlier_returns] = np.nan # remove outlier data
		self._ffill_and_dropna(self.df, 50) # removed discontinued etfs whose data is ffill

	def _ffill_and_dropna(self, df, limit=None):
		old_shape = df.shape[1]
		df.fillna(method='ffill', inplace=True, limit=limit)
		df.dropna(axis='columns', inplace=True)
		self.Log(f'{old_shape} to {df.shape[1]} columns - {old_shape - df.shape[1]} NA columns dropped')
	def symbol_helper(self, id):
		return self.qb.Symbol(id).Value

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

		return -1, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

	def get_pairs_data(self, pair_tickers):
		failed_count = [0]*4

		pairs_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [] }
		for pair in pair_tickers:
			stock1, stock2 = pair[0], pair[1]
			failed_idx, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period = self.stationarity_check(self.df, stock1, stock2)
			if failed_idx != -1:
				failed_count[failed_idx] += 1
				continue

			pairs_data['Stock1'].append(stock1)
			pairs_data['Stock2'].append(stock2)
			pairs_data['Beta'].append(beta)
			pairs_data['p'].append(p_value)
			pairs_data['H'].append(H)
			pairs_data['Half-life'].append(half_life)
			pairs_data['Avg zero cross period'].append(int(avg_cross_period))

			assert abs(residual.mean()) <= 1e-9

		pairs = pd.DataFrame(pairs_data)

		self.Log(f'{failed_count[0]} failed cointegration test')
		self.Log(f'{failed_count[1]} failed H exp criterion')
		self.Log(f'{failed_count[2]} failed half-life criterion')
		self.Log(f'{failed_count[3]} failed avg zero cross period criterion')

		self.Log(f'Found {pairs.shape[0]} pairs')
		self.Log(pairs)

		return pairs

	def get_best_z_thresholds(self, pairs, initial_capital=1000):
		pairs_mean = {}
		pairs_std = {}
		pairs_z_threshold = {}

		for stock1, stock2, beta, p, H, half_life, avg_cross_period in pairs.values:
			# self.Log(f'Simulating pair [{self.symbol_helper(stock1)}-{self.symbol_helper(stock2)}]')
			pair_key = stock1 + '-' + stock2

			pair_df = self.validation_df.loc[:, [stock1, stock2]]
			training_pair_df = self.df.loc[:, [stock1, stock2]]

			pair_df_spread = pair_df[stock1] - beta * pair_df[stock2]
			training_pair_df_spread = training_pair_df[stock1] - beta * training_pair_df[stock2]

			mean = np.mean(training_pair_df_spread)
			std = np.std(training_pair_df_spread)

			# center spread 
			pair_df['z'] = (pair_df_spread - mean) / std

			pairs_mean[pair_key] = mean
			pairs_std[pair_key] = std

			best_final_pnl = -1e9
			best_entry_z_threshold, best_exit_z_threshold = None, None
			best_position, best_fees, best_margin = None, None, None
			
			for entry_z_threshold in np.linspace(1.0, 3.0, 20):
				for exit_z_threshold in np.linspace(0.0, 1.0, 10):
					position, margin, fees = self.backtest(pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold)
					final_pnl = margin[-1]
					if final_pnl > best_final_pnl:
						best_final_pnl = final_pnl
						best_entry_z_threshold, best_exit_z_threshold = entry_z_threshold, exit_z_threshold
						best_position = position
						best_margin = margin
						best_fees = fees
			
			pairs_z_threshold[pair_key] = (best_entry_z_threshold, best_exit_z_threshold)
		
		for pair in pairs_z_threshold:
			self.Log(f'{pair}: ({pairs_z_threshold[pair][0], pairs_z_threshold[pair][1]})')

		return pairs_mean, pairs_std, pairs_z_threshold

	def backtest(self, pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold=2.0, exit_z_threshold=0.5):
		position = { stock1: [0], stock2: [0] }
		capital = initial_capital
		margin = [capital]
		fees = [(0, 0, 0)]

		for time, data_at_time in pair_df.iterrows():
			stock1_close = data_at_time[stock1]
			stock2_close = data_at_time[stock2]
			cur_z_spread = data_at_time['z']

			position_direction = np.sign(position[stock1][-1])

			stock1_shares, stock2_shares = 0, 0
			commission, slippage, short_rental = 0, 0, 0
			if position_direction == 0:
				if (cur_z_spread <= -entry_z_threshold or cur_z_spread >= entry_z_threshold):
					if beta > 1:
						stock2_shares = min(np.floor(capital / stock2_close), np.floor(capital / stock1_close * beta))
						stock1_shares = np.ceil(stock2_shares / beta)
					else:
						stock1_shares = min(np.floor(capital / stock1_close), np.floor(capital / stock2_close / beta))
						stock2_shares = np.ceil(stock1_shares * beta)
					
					assert stock1_shares >= 0
					assert stock2_shares >= 0
						
					is_long = cur_z_spread <= -entry_z_threshold

					position[stock1].append(stock1_shares if is_long else -stock1_shares)
					position[stock2].append(-stock2_shares if is_long else stock2_shares)
					pos_stock1, pos_stock2 = position[stock1][-1], position[stock2][-1]

					portfolio_value = pos_stock1 * stock1_close + pos_stock2 * stock2_close
					commission = 0.0008 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					slippage = 0.0020 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					capital -= slippage + commission
					capital -= portfolio_value
					assert capital >= 0, (commission, slippage, portfolio_value, pos_stock1*stock1_close, pos_stock2*stock2_close, pos_stock1, pos_stock2, stock1_close, stock2_close, beta)
				else:
					position[stock1].append(0)
					position[stock2].append(0)
			else:
				short_rental = -position[stock2][-1] * stock2_close * 0.01/252 if position_direction > 0 else -position[stock1][-1] * stock1_close * 0.01/252
				capital -= short_rental
				if ((position_direction > 0 and cur_z_spread >= exit_z_threshold) or (position_direction < 0 and cur_z_spread <= -exit_z_threshold)):
					pos_stock1, pos_stock2 = position[stock1][-1], position[stock2][-1]
					portfolio_value = pos_stock1 * stock1_close + pos_stock2 * stock2_close
					commission = 0.0008 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					slippage = 0.0020 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					capital -= commission + slippage
					capital += portfolio_value

					position[stock1].append(0)
					position[stock2].append(0)
				else:
					position[stock1].append(position[stock1][-1])
					position[stock2].append(position[stock2][-1])
			
			pos_stock1, pos_stock2 = position[stock1][-1], position[stock2][-1]
			portfolio_value = pos_stock1 * stock1_close + pos_stock2 * stock2_close
			margin.append(capital + portfolio_value) # store margin if liquidated everything at point in time
			fees.append((commission, slippage, short_rental))

		return position, margin, fees

	def OnData(self, data):
		# Update the price series everyday
		new_entry_dict = {}
		time = self.Time
		for serialised_symbol in self.df.columns: # cos of the weird ass symbol naming system
			if data.Bars.ContainsKey(serialised_symbol):
				trade_bar = data[serialised_symbol]
				new_entry_dict[serialised_symbol] = trade_bar.Close
				time = trade_bar.EndTime
			else:
				new_entry_dict[serialised_symbol] = np.nan
		new_entry = pd.DataFrame(new_entry_dict, index=[time])
		self.df = pd.concat([self.df, new_entry])
		self.df.fillna(method='ffill', inplace=True) # some days where ondata gives all nan

		# place orders
		orders = { stock: 0 for stock in self.df.columns }

		cur_data = self.df.iloc[-1]
		for stock1, stock2, beta, p, H, half_life, avg_cross_period in self.pairs.values:
			pair_key = stock1 + '-' + stock2

			stock1_close = cur_data[stock1]
			stock2_close = cur_data[stock2]
			mean = self.pairs_mean[pair_key]
			std = self.pairs_std[pair_key]
			entry_z_threshold, exit_z_threshold = self.pairs_z_threshold[pair_key]

			cur_price_spread = stock1_close - beta * stock2_close
			cur_z_spread = (cur_price_spread - mean) / std

			position_direction = np.sign(self.positions[pair_key][0])

			if position_direction == 0 and (cur_z_spread <= -entry_z_threshold or cur_z_spread >= entry_z_threshold):
				capital = (self.Portfolio.TotalPortfolioValue - 25000) / self.pairs.shape[0]
				if beta > 1:
					stock2_shares = min(np.floor(capital / stock2_close), np.floor(capital / stock1_close * beta))
					stock1_shares = np.ceil(stock2_shares / beta)
				else:
					stock1_shares = min(np.floor(capital / stock1_close), np.floor(capital / stock2_close / beta))
					stock2_shares = np.ceil(stock1_shares * beta)
				
				assert stock1_shares >= 0
				assert stock2_shares >= 0
					
				is_long = cur_z_spread <= -entry_z_threshold

				stock1_new_position = stock1_shares if is_long else -stock1_shares
				stock2_new_position = -stock2_shares if is_long else stock2_shares

				orders[stock1] += stock1_new_position
				orders[stock2] += stock2_new_position
				self.positions[pair_key] = (stock1_new_position, stock2_new_position)
			
			elif ((position_direction > 0 and cur_z_spread >= exit_z_threshold) or (position_direction < 0 and cur_z_spread <= -exit_z_threshold)):
				stock1_old_position, stock2_old_position = self.positions[pair_key]

				orders[stock1] -= stock1_old_position
				orders[stock2] -= stock2_old_position
				self.positions[pair_key] = (0, 0)

		for symbol in orders:
			if orders[symbol] != 0:
				self.MarketOrder(symbol, orders[symbol])

# to do, make std, mean, and beta dynamic 
# using rolling window or kalman filter?

#