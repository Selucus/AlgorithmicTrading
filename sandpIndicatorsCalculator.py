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
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)




df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date)

df = df.stack()



df.index.names = ['date','ticker']
df.columns = df.columns.str.lower()


df['garman_klass_volatility'] = ((np.log(df['high'])-np.log(df['low']))**2)/2 - (2*np.log(2)-1)*((np.log(df['adj close'])) - np.log(df['open']))**2

# group by ticker and calculate relative strength index
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

# look at using other bolinger band levels
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])

df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])

df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])

# using ATR - average true range

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length = 14)

    #normalises
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

# moving average convergence / divergence
def compute_macd(close):
    macd = pandas_ta.macd(close=close,length=20).iloc[:,0]

    #normalises for machine learning
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)


# dollar volume in millions

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6






with open('df.pkl','wb') as f:
    pickle.dump(df,f)
    f.close()
"""
with open('df.pkl', 'rb') as f:

    df = pickle.load(f) # deserialize using load()


# to reduce training time with features and strategies we can aggregate to a monthly level 
# and filter only top 150 most liquid stocks

last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume','volume','open','high','low','close']]


data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
           df.unstack()[last_cols].resample('M').last().stack('ticker')],
           axis=1)).dropna()

# calculate 5-year rolling average of dollar volume for each stocks before filtering

data['dollar_volume'] = (data.loc[:,'dollar_volume'].unstack('ticker').rolling(5*12,min_periods=12).mean().stack())

data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume','dollar_vol_rank'],axis=1)


# calculate monthly returns for different time horizons as features

def calculate_returns(df):


    outlier_cutoff = 0.005

    lags = [1,2,3,6,9,12]

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                            .pct_change(lag)
                            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                                                    upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1)
                                )
    return df


data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

# download fama french factors to estimate exposure of assets to common risk factors

# look at not dropping risk factor - RF
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
               'famafrench',
               start='2010')[0].drop('RF',axis=1)

factor_data.index = factor_data.index.to_timestamp()

# turn percentages into numbers
factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

factor_data = factor_data.join(data['return_1m']).sort_index()


# filter out stocks with less than 10 months data
observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

# calculate rolling factor betas

betas = (factor_data.groupby(level=1,group_keys=False).apply(lambda x: RollingOLS(endog=x['return_1m'],exog=sm.add_constant(x.drop('return_1m',axis=1)),window=min(24,x.shape[0]),
                                                                           min_nobs=len(x.columns)+1).fit(params_only=True).params.drop('const',axis=1)))

factors = ['Mkt-RF','SMB','HML','RMW','CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:,factors] = data.groupby('ticker', group_keys = False)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()

with open('data.pkl','wb') as f:
    pickle.dump(data,f)
    f.close()


