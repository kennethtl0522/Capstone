import yfinance as yf
import pandas as pd

data = yf.Ticker("NDAQ").history(period="6mo", interval="1d")
df = pd.DataFrame(data)

features = df[['Close']]
features.to_csv("./stock/features.csv")