import os
import numpy as np
import pandas as pd
import datetime
from utils import ffill_and_dropna

# note alpaca's data is dog shit because its only from one exchange --> meaning it excludes a lot of data completely
# if i'm not wrong, seems like alpaca stops you from getting data the moment the stock is delisted 
# this means all the data i'm using has survivorship bias

# ok for now, because i'm just trying to implement the strategy first, then i'll try with quantconnnect's data
class AlpacaUtils:
	def get_data(self, tickers, start_date, end_date, resolution):
		display('Is resolution and start and end date as pulled data agreeable?')

		directory = './alpaca_data/bars/day'
		close_df, volume_df = pd.DataFrame({}), pd.DataFrame({})
		tickers_set = set(tickers)
		max_n1, max_n2 = 0, 0
		no_data_found = []

		for i, filename in enumerate(os.listdir(directory)):
			symbol = '.'.join(filename.split('.')[:-1])
			
			if len(tickers) == 0 or symbol in tickers_set:
				df = pd.read_csv(directory+'/'+filename, index_col='timestamp')
				# pd.Timestamp(f'2016-01-01', tz=NY).date().isoformat()
				df.index = pd.to_datetime(df.index).date
				
				close = df[['close']].rename(columns={'close': symbol})
				volume = df[['volume']].rename(columns={'volume': symbol})

				max_n1 = max(len(close), max_n1)
				max_n2 = max(len(volume), max_n2)

				close_df = close_df.join(close, how='outer')
				volume_df = volume_df.join(volume, how='outer')

		assert max_n1 == close_df.shape[0]
		assert max_n2 == volume_df.shape[0]
		
		display(f'Wanted {len(tickers)} tickers but got {close_df.shape[1]} tickers')
		display(close_df)

		return close_df, volume_df


class DataPipeline(AlpacaUtils):
	def __init__(self, tickers, start_date, validation_start_date, testing_start_date, end_date, resolution='D'):
		self.start_date = datetime.datetime(*start_date).date()
		self.end_date = datetime.datetime(*end_date).date()
		self.validation_start_date = datetime.datetime(*validation_start_date).date()
		self.testing_start_date = datetime.datetime(*testing_start_date).date()
		self.resolution = resolution
		self.original_tickers = tickers

		self.df, self.volume_df = self.get_data(self.original_tickers, self.start_date, self.end_date, self.resolution) # close price

	def preprocess_and_split_data(self, min_avg_volume=10000, min_avg_price=5, limit=7, percent=0.8):
		self.raw_training_df = self.df.loc[self.df.index < self.validation_start_date]
		self.validation_df = self.df.loc[(self.df.index >= self.validation_start_date) & (self.df.index < self.testing_start_date)]
		self.testing_df = self.df.loc[self.df.index >= self.testing_start_date]
		self.training_and_validation_df = self.df.loc[self.df.index < self.testing_start_date]

		self.raw_training_volume_df = self.volume_df.loc[self.volume_df.index < self.validation_start_date]

		# process shit only after splitting! doing before will introduce 
		# 1. survivorship bias due to removing bad cases
		# 2. look-ahead bias by removing stocks that became shit in future

		# don't need to remove ffilled data unless want to remove in raw training df / validation df later on before getting test pairs

		# filter by volume and price
		display('##### Filtering #####')
		filtered_by_volume = self.raw_training_df.loc[:, self.raw_training_volume_df.mean() > min_avg_volume]
		filtered_by_price = filtered_by_volume.loc[:, self.raw_training_df.mean() > min_avg_price]

		display('##### Coordinating #####')
		# coordinate start timings for training data
		self.training_df = self.coordinate_start_timings(filtered_by_price.copy(), limit, percent) 

		display(f'{len(self.original_tickers)} original tickers to {self.training_df.shape[1]} tickers')
		display(f'{self.training_df.shape[0]} + {self.validation_df.shape[0]} + {self.testing_df.shape[0]}') 

		return self.training_df, self.validation_df, self.testing_df, self.training_and_validation_df

	def coordinate_start_timings(self, df, limit, percent):
		# need to coordinate start timings for etfs because some were created later
		display(df.shape)
		old_num_rows = df.shape[0]
		dropped_na_columns = ffill_and_dropna(df, limit=limit, thresh=int(percent*old_num_rows)) 

		idx_to_start = df.notnull().all(axis=1).argmax() # first common non na value

		display(f'Df new start date {df.index[idx_to_start]}, removed first {idx_to_start} or {idx_to_start/old_num_rows*100:.2f}% rows')

		return df.iloc[idx_to_start:]
