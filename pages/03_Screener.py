import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
from functions import *
from millify import millify


st.title('Screener')

csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(0, len(symbol)):
    symbol[i] = symbol[i] + ".NS"

st.write("A stock screener is a set of tools that allow investors to quickly sort through various parameters of the stock")
ticker_input = st.selectbox('Enter or Choose stock', symbol)
start_input = dt.datetime.today() - dt.timedelta(120)
end_input = dt.datetime.today()

df = yf.download(ticker_input,start_input,end_input)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date


stock = yf.Ticker(ticker_input)
info = stock.info
closing_price = round((df['Close'].iloc[-1:]),2)
opening_price = round(df['Open'].iloc[-1:],2).astype('str')
sma_df = calc_moving_average(df,12)
sma_df_tail = round(sma_df['sma'].iloc[-1:].astype('int64'),2)
ema_df_tail = round(sma_df['ema'].iloc[-1:].astype('int64'),2)

macd_df = calc_macd(df)
ema26_df_tail = round(macd_df['ema26'].iloc[-1:].astype('int64'),2)
macd_df_tail = round(macd_df['macd'].iloc[-1:].astype('int64'), 2)
signal_df_tail = round(macd_df['signal'].iloc[-1:].astype('int64'), 2)

rsi_df = RSI(df,14)
rsi_df_tail = round(rsi_df['RSI'].iloc[-1:].astype('int64'), 2)

adx_df = ADX(df,14)
adx_df_tail = round(adx_df.iloc[-1:].astype('int64'), 2)

breaking_out = is_breaking_out(df)
consolidating = is_consolidating(df)

rsi_df = RSI(df,14)
rsi_df_tail = round(rsi_df['RSI'].iloc[-1:].astype('int64'), 2)

adx_df = ADX(df,14)
adx_df_tail = round(adx_df.iloc[-1:].astype('int64'), 2)


# metrics at a glance
with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('52 Week Low', millify(info['fiftyTwoWeekLow'],2))
        col2.metric('52 Week High', millify(info['fiftyTwoWeekHigh'],2))

        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric('Market Day Low', millify(info['regularMarketDayLow'], 2))
        col_2.metric('Market Day High', millify(info['regularMarketDayHigh'], 2))

with st.container():
        co_1, co_2,co_3,co_4 = st.columns(4)
        co_1.metric('EBITDA Margin', info['ebitdaMargins'])
        co_2.metric('Profit Margin', info['profitMargins'])
        co_3.metric('Gross Margin', info['grossMargins'])
        co_4.metric('Operating Margin', info['operatingMargins'])

with st.container():
        co_11, co_22,co_33,co_44 = st.columns(4)
        co_11.metric('Current Ratio', info['currentRatio'])
        co_22.metric('Return on Assets', info['returnOnAssets'])
        co_33.metric('Debt to Equity', info['debtToEquity'])
        co_44.metric('Return on Equity', info['returnOnEquity'])

with st.container():
        c_1, c_2,c_3,c_4 = st.columns(4)
        c_1.metric('Closing Price', millify(closing_price, precision=2))
        c_2.metric('Simple Moving Average', millify(sma_df_tail, precision=2))
        c_3.metric('Exponential Moving Average', millify(ema_df_tail, precision=2))
        c_4.metric('Exponential Moving Average over period 26', millify(ema26_df_tail,2))

with st.container():
        c_11, c_22,c_33,c_44 = st.columns(4)
        c_11.metric('Relative Strength Index', rsi_df_tail)
        c_22.metric('Average Directional Index', adx_df_tail)
        c_33.metric('MACD', macd_df_tail)
        c_44.metric('Signal', signal_df_tail)

with st.container():
        cc_11, cc_22,cc_33,cc_44 = st.columns(4)
        cc_22.metric('Breaking Out??', breaking_out)
        cc_11.metric('Consolidating??', consolidating)
        cc_33.metric('50 Day Average', millify(info['fiftyDayAverage'],2))
        cc_44.metric('Recommendation', info['recommendationKey'].upper())
