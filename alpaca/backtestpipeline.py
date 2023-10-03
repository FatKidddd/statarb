import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

class BacktestPipeline():
	def __init__(self, percent_margin_buffer=0.1):
		self.percent_margin_buffer = percent_margin_buffer

	def prepare_training_and_testing_df(self, training_df, testing_df, stock1, stock2, beta):
		training_pair_df = training_df.loc[:, [stock1, stock2]]
		pair_df = testing_df.loc[:, [stock1, stock2]]

		training_pair_df_spread = training_pair_df[stock1] - beta * training_pair_df[stock2]
		pair_df_spread = pair_df[stock1] - beta * pair_df[stock2]

		mean = np.mean(training_pair_df_spread)
		std = np.std(training_pair_df_spread)

		training_pair_df['z'] = (training_pair_df_spread - mean) / std
		pair_df['z'] = (pair_df_spread - mean) / std

		return training_pair_df, pair_df

	def validation_backtest(self, training_df, validation_df, pairs, initial_capital=1000):
		validation_backtest_results = { stock1+'-'+stock2: {} for stock1, stock2 in pairs[['Stock1', 'Stock2']].values }

		for entry_z_threshold in np.linspace(1.0, 2.5, 5):
			for exit_z_threshold in np.linspace(0.0, 1.0, 4):
				for stock1, stock2, beta, p, H, half_life, avg_cross_period in pairs.values:
					# display(f'Simulating pair [{stock1}-{stock2}]')
					training_pair_df, pair_df = self.prepare_training_and_testing_df(training_df, validation_df, stock1, stock2, beta)

					position, fees, margin = self.backtest(pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold)

					past_margin = validation_backtest_results[stock1+'-'+stock2].get('margin')
					if past_margin == None or margin[-1] > past_margin[-1]:
						validation_backtest_results[stock1+'-'+stock2] = {
							'training_pair_df': training_pair_df,
							'pair_df': pair_df,
							'beta': beta,
							'entry_z_threshold': entry_z_threshold,
							'exit_z_threshold': exit_z_threshold,
							'position': position,
							'margin': margin,
							'fees': fees
						}

		return validation_backtest_results

	def test_backtest(self, training_and_validation_df, testing_df, test_pairs, validation_backtest_results, initial_capital=1000):
		test_backtest_results = { stock1+'-'+stock2: {} for stock1, stock2 in test_pairs[['Stock1', 'Stock2']].values }

		for stock1, stock2, beta, p, H, half_life, avg_cross_period in test_pairs.values:
			pair_dict = validation_backtest_results[stock1+'-'+stock2]

			entry_z_threshold = pair_dict['entry_z_threshold']
			exit_z_threshold = pair_dict['exit_z_threshold']

			# display(f'Simulating pair [{stock1}-{stock2}]')
			training_pair_df, pair_df = self.prepare_training_and_testing_df(training_and_validation_df, testing_df, stock1, stock2, beta)

			position, fees, margin = self.backtest(pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold)

			test_backtest_results[stock1+'-'+stock2] = {
				'training_pair_df': training_pair_df,
				'pair_df': pair_df,
				'beta': beta,
				'entry_z_threshold': entry_z_threshold,
				'exit_z_threshold': exit_z_threshold,
				'position': position,
				'margin': margin,
				'fees': fees
			}

		return test_backtest_results

	def backtest(self, pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold):
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

			usable_capital = capital * (1-self.percent_margin_buffer)
			if position_direction == 0:
				if (cur_z_spread <= -entry_z_threshold or cur_z_spread >= entry_z_threshold):
					# adding the / 2 to avoid margin calls??? im p sure this isnt right tho
					if beta > 1:
						stock2_shares = min(np.floor(usable_capital / stock2_close / 2), np.floor(usable_capital / stock1_close * beta / 2))
						stock1_shares = np.ceil(stock2_shares / beta)
					else:
						stock1_shares = min(np.floor(usable_capital / stock1_close / 2), np.floor(usable_capital / stock2_close / beta / 2))
						stock2_shares = np.ceil(stock1_shares * beta)
					
					assert stock1_shares > 0
					assert stock2_shares > 0
						
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

		return position, fees, margin

	def plot_pair_backtest(self, results, pair_key):
		stock1, stock2 = pair_key.split('-')

		pair_dict = results[pair_key]

		training_pair_df = pair_dict['training_pair_df']
		pair_df = pair_dict['pair_df']
		beta = pair_dict['beta']
		entry_z_threshold = pair_dict['entry_z_threshold']
		exit_z_threshold = pair_dict['exit_z_threshold']
		position	= pair_dict['position']
		margin	= pair_dict['margin']
		fees	= pair_dict['fees']

		display(f'[{stock1} {stock2}] Entry z threshold: {entry_z_threshold:.3f} Exit z threshold: {exit_z_threshold:.3f} Cum PnL: {(margin[-1]-margin[0]):.3f}')

		plt.figure(figsize =(12, 5))
		G = gridspec.GridSpec(2, 3)
		ax1 = plt.subplot(G[0, 0])
		ax2 = plt.subplot(G[0, 1])
		ax3 = plt.subplot(G[0, 2])
		ax4 = plt.subplot(G[1, 0])
		ax5 = plt.subplot(G[1, 1])
		ax6 = plt.subplot(G[1, 2])

		stock1_symbol = stock1
		stock2_symbol = stock2

		time = np.concatenate([training_pair_df.index, pair_df.index])

		# plot pairs individual price
		ax1.plot(time, np.concatenate([training_pair_df[stock1], pair_df[stock1]]), 'r', label=stock1_symbol)
		ax1.plot(time, np.concatenate([training_pair_df[stock2], pair_df[stock2]]), 'g', label=stock2_symbol)
		ax1.axvline(x=pair_df.index[0], ymin=0, ymax=1, linewidth=1, color='b')
		ax1.set_xlabel('Date')
		ax1.set_title('Close price pair comparison')
		ax1.legend()

		# plot z spread price
		ax2.plot(time, np.concatenate([training_pair_df['z'], pair_df['z']]), 'c', label=f'z = norm {stock1_symbol}-{beta:.2f}*{stock2_symbol}')
		ax2.axvline(x=pair_df.index[0], ymin=0, ymax=1, linewidth=1, color='b')

		pos_stock1 = np.array(position[stock1])[1:]
		pos_stock2 = np.array(position[stock2])[1:]
		np_margin = np.array(margin)

		long_indices = pos_stock1 > 0
		short_indices = pos_stock1 < 0
		ax2.plot(pair_df.index[long_indices], pair_df.loc[long_indices, 'z'], 'g.')
		ax2.plot(pair_df.index[short_indices], pair_df.loc[short_indices, 'z'], 'r.')
		ax2.axhline(y=entry_z_threshold, xmin=0, xmax=1, linewidth=1, color='m')
		ax2.axhline(y=-entry_z_threshold, xmin=0, xmax=1, linewidth=1, color='m')
		ax2.axhline(y=exit_z_threshold, xmin=0, xmax=1, linewidth=1, color='brown')
		ax2.axhline(y=-exit_z_threshold, xmin=0, xmax=1, linewidth=1, color='brown')
		ax2.set_xlabel('Date')
		ax2.set_title('z')
		ax2.legend()

		# margin, fees, and drawdown got one extra starting pt
		# plot cumulative pnl
		ax3.plot(np_margin)
		ax3.set_title('Margin')

		# plot fees
		np_fees = np.array(fees)
		cumsum_fees = np.cumsum(np_fees, axis=0)
		cumsum_total_fees = np.sum(cumsum_fees, axis=1)
		ax4.plot(cumsum_fees[:, 0], label='commission')
		ax4.plot(cumsum_fees[:, 1], label='slippage')
		ax4.plot(cumsum_fees[:, 2], label='short rental')
		ax4.plot(cumsum_total_fees, label='total')
		ax4.set_title('Fees')
		ax4.legend()

		# plot exposure 
		stock1_invested = pair_df[stock1] * pos_stock1
		stock2_invested = pair_df[stock2] * pos_stock2
		net_exposure = stock1_invested + stock2_invested
		abs_exposure = np.absolute(stock1_invested) + np.absolute(stock2_invested)
		ax5.plot(stock1_invested, label='stock1')
		ax5.plot(stock2_invested, label='stock2')
		ax5.plot(net_exposure, label='net exposure')
		ax5.plot(abs_exposure, label='abs exposure')
		ax5.set_title('Exposure')
		ax5.legend()

		# plot drawdown
		cumret = np_margin / np_margin[0] - 1
		highwatermark=np.zeros(cumret.shape)
		drawdown=np.zeros(cumret.shape)
		drawdownduration=np.zeros(cumret.shape)
		
		for t in np.arange(1, cumret.shape[0]):
			highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
			drawdown[t]=(1+cumret[t])/(1+highwatermark[t])-1
			if drawdown[t]==0:
				drawdownduration[t]=0
			else:
				drawdownduration[t]=drawdownduration[t-1]+1
				
		maxDD, i=np.min(drawdown), np.argmin(drawdown) # drawdown < 0 always
		maxDDD=np.max(drawdownduration)
		
		ax6.plot(drawdown)
		ax6.set_title('Drawdown')

		plt.tight_layout()
		plt.show()
	
	


