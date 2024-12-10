import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

nasdaq = yf.Ticker("NDAQ").history(period="6mo", interval="1d")
nasdaq = pd.DataFrame(nasdaq)