import streamlit as st

#streamlit page config
st.set_page_config(page_title="Predict Stocks", layout="wide", page_icon=":chart_with_upwards_trend:")
hide_menu_style ="""
        <style>
        footer{visibility:hidden;}
        </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

#feature 1 - Home

st.markdown(" ### Predict Stocks is the all in one financial website for the retail investors where retail investors can take a look at all Fundamental Information, Technical Indicators, Screeners, Pattern Recognition and Next-Day Forecasting of all the National Stock Exchange (NSE) listed stocks.")
st.markdown('#### Features of the Machine Learning based Web-app:')
st.markdown('##### -  One can gain information of all the stocks that are listed on National Stock Exchange (NSE) (To Be Exact - 1773 Companies). ')
st.markdown('##### -  Webapp have features like Fundamental Information, Technical Indicators, Screener, Pattern Recognition, Next-Day Forecasting (Machine Learning based).')

st.markdown('#### Details related to stock prices:')
st.markdown('#####  - All the historical prices for last 10 years are taken from Yahoo Finance and then further pushed to SQLIte database.')
st.markdown('##### - Time period for stock prices is one day.')

st.markdown('#### Fundamental Information feature overview:')
st.markdown(' ##### -  It contains all the information related to company. ')
st.markdown(' ##### -  Historical prices of the company of last 10 years with candlestick and line chart.')
st.markdown(' ##### -  One can download historical prices too.')
st.markdown(' ##### - All quarterly and annually Financial Results, Balance Sheet, Cash Flow and Splits & Dividends')


st.markdown('#### Technical Indicators feature overview:')
st.markdown(' ##### -  It contains all the famous indicators that traders and investors use while investing.')
st.markdown(' ##### -  More than 11 indicators are there.')

st.markdown('#### Technical Screener feature overview:')
st.markdown(' ##### -  It contains all the important parameters or metrics that are related to company.')
st.markdown(' ##### -  It also contains one more important feature, that is it give signals whether selected stock is breaking out or not. Traders do this manually by looking at candles.')
st.markdown(' ##### -  With breaking out it also gives signals for consolidating or not.')

st.markdown('#### Pattern Recognition feature overview:')
st.markdown(' ##### -  It is important feature of the webapp.')
st.markdown(' ##### -  Traders do this manually by looking at candles.')
st.markdown(' ##### - We automate this thing by scanning all the candlestick patterns for the selected stock, then it generates signals whether it is bullish or bearish')

st.markdown('#### Next-Day Forecasting feature overview:')

st.markdown(' ##### -  We have build the efficient Machine Learning model to predict the next day price.')
st.markdown(' ##### -  Our model trained on past 5 years of historical data and while predicting it looks for past 2 months to predict next-day price.')

