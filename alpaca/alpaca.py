from enum import Enum
import time
import alpaca_trade_api as tradeapi
import asyncio
import os
import pandas as pd
import sys
from alpaca_trade_api.rest import TimeFrame, URL
from alpaca_trade_api.rest_async import gather_with_concurrency, AsyncRest

NY = 'America/New_York'

class DataType(str, Enum):
  Bars = "Bars"
  Trades = "Trades"
  Quotes = "Quotes"



def get_data_method(data_type: DataType):
  if data_type == DataType.Bars:
    return rest.get_bars_async
  elif data_type == DataType.Trades:
    return rest.get_trades_async
  elif data_type == DataType.Quotes:
    return rest.get_quotes_async
  else:
    raise Exception(f"Unsupoported data type: {data_type}")


async def get_historic_data_base(symbols, data_type: DataType, label, start, end,
																 timeframe: TimeFrame = None):
  """
  base function to use with all
  :param symbols:
  :param start:
  :param end:
  :param timeframe:
  :return:
  """
  major = sys.version_info.major
  minor = sys.version_info.minor
  if major < 3 or minor < 6:
    raise Exception('asyncio is not support in your python version')
  msg = f"Getting {data_type} data for {len(symbols)} symbols"
  msg += f", timeframe: {timeframe}" if timeframe else ""
  msg += f" between dates: start={start}, end={end}"
  print(msg)
  step_size = 1000
  results = []
  for i in range(0, len(symbols), step_size):
    tasks = []
    for symbol in symbols[i:i+step_size]:
      args = [symbol, start, end, timeframe.value] if timeframe else \
          [symbol, start, end]
      tasks.append(get_data_method(data_type)(*args, limit=10000, adjustment='all'))

    if minor >= 8:
      results.extend(await asyncio.gather(*tasks, return_exceptions=True))
    else:
      results.extend(await gather_with_concurrency(500, *tasks))

  bad_requests = 0
  for response in results:
    if isinstance(response, Exception):
      print(f"Got an error: {response}")
    elif not len(response[1]):
      bad_requests += 1
    else:
      symbol = response[0] 
      df = response[1] 
      print(symbol, df.shape)
      period = 'hour' if timeframe == TimeFrame.Hour else 'day'
      df.to_csv(f'./alpaca_data/{label}/{period}/{symbol}.csv')


  print(f"Total of {len(results)} {data_type}, and {bad_requests} empty responses.")


async def get_historic_bars(symbols, start, end, timeframe: TimeFrame):
  await get_historic_data_base(symbols, DataType.Bars, 'bars', start, end, timeframe)


async def get_historic_trades(symbols, start, end, timeframe: TimeFrame):
  await get_historic_data_base(symbols, DataType.Trades, 'trades', start, end)


async def get_historic_quotes(symbols, start, end, timeframe: TimeFrame):
  await get_historic_data_base(symbols, DataType.Quotes, 'quotes', start, end)


async def main(symbols):
  # start = pd.Timestamp(f'2018-01-01', tz=NY).date().isoformat()
  # end = pd.Timestamp(f'2022-12-31', tz=NY).date().isoformat()
  # timeframe: TimeFrame = TimeFrame.Hour
  start = pd.Timestamp(f'2016-01-01', tz=NY).date().isoformat()
  end = pd.Timestamp(f'2022-12-31', tz=NY).date().isoformat()
  timeframe: TimeFrame = TimeFrame.Day
  # await get_historic_bars(symbols, start, end, timeframe)
  # await get_historic_trades(symbols, start, end, timeframe)
  await get_historic_quotes(symbols, start, end, timeframe)


if __name__ == '__main__':
  # Set your Alpaca API credentials
  api_key_id = 'PKD10XAWJ1BMO8W1AEFQ'
  api_secret = 'PliuHgzss3Ad4SPzScq4n5KJ7Ld5WGjaurZHGz6E'
  base_url = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading
  feed = "iex"  # change to "sip" if you have a paid account

  rest = AsyncRest(key_id=api_key_id,
                    secret_key=api_secret)

  api = tradeapi.REST(key_id=api_key_id,
                      secret_key=api_secret,
                      base_url=URL(base_url))

  start_time = time.time()

  symbols = [el.symbol for el in api.list_assets(status='active')]
  # symbols = symbols[:5]

  asyncio.run(main(symbols))
  print(f"took {time.time() - start_time} sec")