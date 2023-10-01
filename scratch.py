import requests

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=GSJY&interval=60min&apikey=UG1PR248C2N23C29'
r = requests.get(url)
data = r.json()

print(data)