from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
import pickle
warnings.filterwarnings('ignore')

"""
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.','-')

symbols_list = sp500['Symbol'].unique().tolist()

end_date = '2024-09-06'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365)




df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date)

df = df.stack()
with open('df.pkl','wb') as f:
    pickle.dump(df,f)
    f.close()
"""
with open('df.pkl', 'rb') as f:

    df = pickle.load(f) # deserialize using load()

df.index.names = ['date','ticker']
df.columns = df.columns.str.lower()

print(df)