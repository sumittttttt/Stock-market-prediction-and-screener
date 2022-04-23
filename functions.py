import streamlit as st
import numpy as np



@st.cache
def calc_moving_average(data, size):
        df = data.copy()
        df['sma'] = df['Close'].rolling(int(size)).mean()
        df['ema'] = df['Close'].ewm(span=size, min_periods=size).mean()
        df.dropna(inplace=True)
        return df

#Function for Moving Average Convergence Divergence
@st.cache
def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df

#function for bollinger Bands
@st.cache
def calc_bollinger(data, size):
        df = data.copy()
        df['sma'] = df['Close'].rolling(int(size)).mean()
        df["bolu"] = df["sma"] + 2 * df['Adj Close'].rolling(int(size)).std(ddof=0)
        df["bold"] = df["sma"] - 2 * df['Adj Close'].rolling(int(size)).std(ddof=0)
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df

#function for ATR-Average True Range
@st.cache
def ATR(data, n):
        "function to calculate True Range and Average True Range"
        df = data.copy()
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        df['ATR'] = df['TR'].rolling(n).mean()
        df2 = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
        return df2

    #function to calculate RSI
@st.cache
def RSI(data, n):
        "function to calculate RSI"
        df = data.copy()
        df['delta'] = df['Adj Close'] - df['Adj Close'].shift(1)
        df['gain'] = np.where(df['delta'] >= 0, df['delta'], 0)
        df['loss'] = np.where(df['delta'] < 0, abs(df['delta']), 0)
        avg_gain = []
        avg_loss = []
        gain = df['gain'].tolist()
        loss = df['loss'].tolist()
        for i in range(len(df)):
            if i < n:
                avg_gain.append(np.NaN)
                avg_loss.append(np.NaN)
            elif i == n:
                avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
                avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
            elif i > n:
                avg_gain.append(((n - 1) * avg_gain[i - 1] + gain[i]) / n)
                avg_loss.append(((n - 1) * avg_loss[i - 1] + loss[i]) / n)
        df['avg_gain'] = np.array(avg_gain)
        df['avg_loss'] = np.array(avg_loss)
        df['RS'] = df['avg_gain'] / df['avg_loss']
        df['RSI'] = 100 - (100 / (1 + df['RS']))
        return df

#Function to calculate ADX
@st.cache
def ADX(data, n):
        "function to calculate ADX"
        df2 = data.copy()
        df2['TR'] = ATR(df2, n)[
            'TR']  # the period parameter of ATR function does not matter because period does not influence TR calculation
        df2['DMplus'] = np.where((df2['High'] - df2['High'].shift(1)) > (df2['Low'].shift(1) - df2['Low']),
                                 df2['High'] - df2['High'].shift(1), 0)
        df2['DMplus'] = np.where(df2['DMplus'] < 0, 0, df2['DMplus'])
        df2['DMminus'] = np.where((df2['Low'].shift(1) - df2['Low']) > (df2['High'] - df2['High'].shift(1)),
                                  df2['Low'].shift(1) - df2['Low'], 0)
        df2['DMminus'] = np.where(df2['DMminus'] < 0, 0, df2['DMminus'])
        TRn = []
        DMplusN = []
        DMminusN = []
        TR = df2['TR'].tolist()
        DMplus = df2['DMplus'].tolist()
        DMminus = df2['DMminus'].tolist()
        for i in range(len(df2)):
            if i < n:
                TRn.append(np.NaN)
                DMplusN.append(np.NaN)
                DMminusN.append(np.NaN)
            elif i == n:
                TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
                DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
                DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
            elif i > n:
                TRn.append(TRn[i - 1] - (TRn[i - 1] / n) + TR[i])
                DMplusN.append(DMplusN[i - 1] - (DMplusN[i - 1] / n) + DMplus[i])
                DMminusN.append(DMminusN[i - 1] - (DMminusN[i - 1] / n) + DMminus[i])
        df2['TRn'] = np.array(TRn)
        df2['DMplusN'] = np.array(DMplusN)
        df2['DMminusN'] = np.array(DMminusN)
        df2['DIplusN'] = 100 * (df2['DMplusN'] / df2['TRn'])
        df2['DIminusN'] = 100 * (df2['DMminusN'] / df2['TRn'])
        df2['DIdiff'] = abs(df2['DIplusN'] - df2['DIminusN'])
        df2['DIsum'] = df2['DIplusN'] + df2['DIminusN']
        df2['DX'] = 100 * (df2['DIdiff'] / df2['DIsum'])
        ADX = []
        DX = df2['DX'].tolist()
        for j in range(len(df2)):
            if j < 2 * n - 1:
                ADX.append(np.NaN)
            elif j == 2 * n - 1:
                ADX.append(df2['DX'][j - n + 1:j + 1].mean())
            elif j > 2 * n - 1:
                ADX.append(((n - 1) * ADX[j - 1] + DX[j]) / n)
        df2['ADX'] = np.array(ADX)
        return df2['ADX']

#function to calculate OBV
@st.cache
def OBV(DF):
        """function to calculate On Balance Volume"""
        df = DF.copy()
        df['daily_ret'] = df['Adj Close'].pct_change()
        df['direction'] = np.where(df['daily_ret'] >= 0, 1, -1)
        df['direction'][0] = 0
        df['vol_adj'] = df['Volume'] * df['direction']
        df['obv'] = df['vol_adj'].cumsum()
        return df['obv']

def is_consolidating(df, percentage=10):
    recent_candlesticks = df[-15:]

    max_close = recent_candlesticks['Close'].max()
    min_close = recent_candlesticks['Close'].min()

    threshold = 1 - (percentage / 100)
    if min_close > (max_close * threshold):
        return 'YES'

    return 'NO'

def is_breaking_out(df, percentage=10):
    last_close = df[-1:]['Close'].values[0]

    if is_consolidating(df[:-1], percentage=percentage):
        recent_closes = df[-16:-1]

        if last_close > recent_closes['Close'].max():
            return 'YES'

    return 'NO'