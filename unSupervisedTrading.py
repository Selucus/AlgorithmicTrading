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


df['garman_klass_volatility'] = ((np.log(df['high'])-np.log(df['low']))**2)/2 - (2*np.log(2)-1)*((np.log(df['adj close'])) - np.log(df['open']))**2

# group by ticker
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
df.xs('AAPL',level=1)['rsi'].plot()
print(df)
