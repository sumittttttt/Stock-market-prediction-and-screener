#webapp python script
#Streamlit webapp
#importing packages

import time
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from finta import TA
import ta
import sqlite3
import talib
from patterns import candlestick_patterns
from functions import *
from millify import millify





#streamlit page config
st.set_page_config(page_title="Predict Stocks", layout="wide", page_icon=":chart_with_upwards_trend:", )
hide_menu_style ="""
        <style>
        footer{visibility:hidden;}
        </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# getting symbols/tickers
csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(0, len(symbol)):
    symbol[i] = symbol[i] + ".NS"

#sidebar features
feature = st.sidebar.radio(
                "Choose feature",
                ('Home','Fundamental Info','Technical Indicators','Screener','Next-Day Forecasting','Pattern Recognition'))

#feature 1 - Home
if(feature =="Home"):
    #st.image("images/logo.png", width=460)
    st.markdown(" ### Predict Stocks is the all in one financial website for the retail investors where retail investors can take a look at all Fundamental Information, Technical Indicators, Screeners, Pattern Recognition and Next-Day Forecasting of all the National Stock Exchange (NSE) listed stocks.")
    st.markdown('#### Features of the Machine Learning based Web-app:')
    st.markdown('##### -  One can gain information of all the stocks that are listed on National Stock Exchange (NSE) (To Be Exact - 1773 Companies). ')
    st.markdown('##### -  Webapp have features like Fundamental Information, Technical Indicators, Screener, Pattern Recognition, Next-Day Forecasting (Machine Learning based).')
    st.markdown('##### -  Database ready.')
    st.markdown('##### -  Mobile responsive webapp.')
    st.markdown('##### -  Webapp comes with Custom Theming and Light and Dark mode.')

    st.markdown('#### Details related to stock prices:')
    st.markdown('#####  - All the historical prices for last 10 years are taken from Yahoo Finance.')
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
    st.markdown(' ##### -  We can say it is a star feature of the webapp.')
    st.markdown(' ##### -  We have build the efficient Machine Learning model to predict the next day price.')
    st.markdown(' ##### -  Our model trained on past 5 years of historical data and while predicting it looks for past 2 months ton predict next-day price.')



if (feature == "Fundamental Info"):
    st.title('Fundamental Information')
    ticker = st.selectbox(
        'Enter or Choose NSE listed Stock Symbol',
        symbol)
    stock = yf.Ticker(ticker)

#company info
    info = stock.info
    st.subheader(info['longName'])
    st.markdown('**Sector**:' + info['sector'])
    st.markdown('**Industry**: ' + info['industry'])
    st.markdown('**Phone**: ' + info['phone'])
    st.markdown(
        '**Address**: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', ' + info['country'])
    st.markdown('**Website**: ' + info['website'])
    with st.expander('See detailed business summary'):
        st.write(info['longBusinessSummary'])


 # closing price chart
    st.subheader('Stocks Prices')
    st.write("Enter period to check closing price of ", ticker)

    #getting date input
    min_value = dt.datetime.today() - dt.timedelta(10 * 365)
    max_value = dt.datetime.today()

    start_input = st.date_input(
            'Enter starting date',
            value=dt.datetime.today()- dt.timedelta(90),
            min_value=min_value, max_value=max_value, help='Enter the starting date from which you have to look the price'
        )

    end_input =  st.date_input(
            'Enter last date',
            value=dt.datetime.today(),
            min_value=min_value, max_value=max_value, help='Enter the last date till which you have to look the price'
        )

    #retrieving data from database

    df = yf.download(ticker, start_input, end_input)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])


    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    historical_csv = convert_df(df)
    st.download_button(
        label="Download historical data as CSV",
        data=historical_csv,
        file_name='historical_df.csv',
        mime='text/csv',
    )

    #radio button ro switch between style
    chart = st.radio(
    "Choose Style",
    ('Candlestick', 'Line Chart'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if (chart == 'Line Chart'):
    # line chart plot
        fig = go.Figure()
        fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Adj Close'],
            name='Closing price'
        )
    )


        fig.update_layout(
        title = {
                'text': 'Stock Prices of ' + ticker,
               'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
               'yanchor': 'top'}, height = 600, template = 'gridon')
        fig.update_yaxes(tickprefix='₹')
        st.plotly_chart(fig, use_container_width=True)

    if (chart == 'Candlestick'):
        fig = go.Figure()
        fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        )
    )

        fig.update_layout(
        title = {
                'text': 'Stock Prices of ' + ticker,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}, height = 600, template = 'gridon')
        fig.update_yaxes(tickprefix='₹')
        st.plotly_chart(fig, use_container_width=True)

#quarterly results
    st.subheader('Quarterly Result')
    st.write('A quarterly result is a summary or collection of unaudited financial statements, such as balance sheets, income statements, and cash flow statements, issued by companies every quarter (three months).')
    quarterly_results = stock.quarterly_financials
    quarterly_results.columns = quarterly_results.columns.date
    quarterly_results.dropna(axis=0, inplace=True)
    quarterly_results = quarterly_results.astype('int64')
    for i in quarterly_results.columns:
        quarterly_results[i] = quarterly_results.apply(lambda x: "{:,}".format(x[i]), axis=1)
    st.dataframe(quarterly_results.style.highlight_max(axis=1, color='lightgreen'),height=1000, width=2000)


    #profit and loss
    st.subheader('Profit & Loss')
    st.write("A profit and loss (P&L) statement is a annually financial report that provides a summary of a company's revenue, expenses and profit.")
    financials = stock.financials
    financials.columns = financials.columns.date
    financials.dropna(axis=0, inplace=True)
    financials = financials.astype('int64')
    for i in financials.columns:
        financials[i] = financials.apply(lambda x: "{:,}".format(x[i]), axis=1)
    st.dataframe(financials.style.highlight_max(axis=1,color='lightgreen'),height=1000)

    #balance sheet
    st.subheader('Balance Sheet')
    st.write("A balance sheet is a financial statement that reports a company's assets, liabilities, and shareholder equity.")
    balance = stock.balance_sheet
    balance.columns = balance.columns.date
    balance.dropna(axis=0, inplace=True)
    balance = balance.astype('int64')
    for i in balance.columns:
        balance[i] = balance.apply(lambda x: "{:,}".format(x[i]), axis=1)
    st.dataframe(balance.style.highlight_max(axis=1,color='lightgreen'),height=1000)

    #cash flow
    st.subheader('Cash Flows')
    st.write("The term cash flow refers to the net amount of cash and cash equivalents being transferred in and out of a company.")
    cf = stock.cashflow
    cf.columns = cf.columns.date
    cf.dropna(axis=0, inplace=True)
    cf = cf.astype('int64')
    for i in cf.columns:
        cf[i] = cf.apply(lambda x: "{:,}".format(x[i]), axis=1)
    st.dataframe(cf.style.highlight_max(axis=1,color='lightgreen'),height=1000)

    #actions
    st.subheader('Splits & Dividends')
    st.write('')
    actions = stock.actions
    actions.index = actions.index.date
    st.table(actions)


#gettig technical indicators
if (feature == 'Technical Indicators'):

    st.title('Technical Indicators')

    # creating dropdown
    ticker = st.selectbox(
        'Enter or Choose',
        symbol)

    # getting date input
    min_value = dt.datetime.today() - dt.timedelta(10 * 365)
    max_value = dt.datetime.today()

    start_input = st.date_input(
        'Enter starting date',
        value=dt.datetime.today() - dt.timedelta(180),
        min_value=min_value, max_value=max_value, help='Enter the starting date from which you have to look the price'
    )

    end_input = st.date_input(
        'Enter last date',
        value=dt.datetime.today(),
        min_value=min_value, max_value=max_value, help='Enter the last date till which you have to look the price'
    )

    df = yf.download(ticker, start_input, end_input)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])


    #graph template
    temp_style = st.radio(
        "Choose Template Style",
        ('ggplot2', 'seaborn', 'plotly_white','plotly_dark', 'gridon'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    #OVERLAP STUDIES INDICATOR
    st.header('OVERLAY INDICATORS')
    st.write("Technical indicators that use the same scale as prices are plotted over the top of the prices on a stock chart. ")

    st.write('#### Moving Average')
    st.write(
        "Moving averages (MA) are one of the most popular and often-used technical indicators in the financial markets. In simple word, a moving average is an indicator that shows the average value of a stock's price over a period (i.e. 10 days, 50 days, 200 days, etc) and is usually plotted along with the closing price.")

    df_ma = calc_moving_average(df, 14)
    df_ma = df_ma.reset_index()

    figMA = go.Figure()
    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['Close'],
            name="Prices"
        )
    )

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['sma'],
            name='SMA '
        )
    )

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['ema'],
            name='EMA '

        )
    )

    figMA.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0
    ))

    figMA.update_layout(height=600, width=1000, title_text='Closing Price of Stock & Moving Average',
                        template=temp_style)

    st.plotly_chart(figMA, use_container_width=True)

    # hma
    st.subheader('Hull Moving Average (HMA)')
    st.write(
        'The Hull Moving Average (HMA) is a directional trend indicator. It captures the current state of the market and uses recent price action to determine if conditions are bullish or bearish relative to historical data.')

    df['HMA']= TA.HMA(df, 14)

    fig_apo = go.Figure()
    fig_apo.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['HMA'],
            name='HMA'
        )
    )
    fig_apo.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name='Close'
        )
    )

    fig_apo.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='HMA', template=temp_style
    )

    st.plotly_chart(fig_apo, use_container_width=True)

    #bollinger bands
    st.subheader('Bollinger Bands')
    st.write(
        "Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.")

    #calculating bollinger bands
    df_boll = calc_bollinger(df, 20)
    df_boll = df_boll.reset_index()

    figBoll = go.Figure()
    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bolu'],
            name='Upper Band'
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['sma'],
            name='SMA'

        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bold'],
            name="Lower Band"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['Close'],
            name="closing price"
        )
    )

    figBoll.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='Closing Price of Stock & Bollinger Band', template=temp_style
    )

    st.plotly_chart(figBoll, use_container_width=True)

    #kama
    st.subheader('KAMA Indicator')
    st.write(" Kaufman’s Adaptive Moving Average (KAMA) is to identify the general trend of current market price action. Basically, when the KAMA indicator line is moving lower, it indicates the existence of a downtrend. On the other hand, when the KAMA line is moving higher, it shows an uptrend. ")

    kama = ta.momentum.KAMAIndicator(df['Close'],20,2,20)
    df['kama'] = kama.kama()
    df = df.reset_index()
    fig_kama = go.Figure()
    fig_kama.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['kama'],
            name="KAMA"
        )
    )
    fig_kama.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name="Close"
        )
    )

    fig_kama.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='KAMA Indicator', template=temp_style
    )
    st.plotly_chart(fig_kama, use_container_width=True)

    ## Momentum indiactors
    st.header('Momentum Indicators')
    st.write('Technical indicator which shows the trend direction and measures the pace of the price fluctuation by comparing current and past values.')

    st.subheader('Average Directional Index (ADX)')
    st.write(
        "ADX stands for Average Directional Movement Index and can be used to help measure the overall strength of a trend. Indicator suggests that a strong trend is present when ADX is above 25 and no trend is present when below 20.")

    df['ADX'] = ADX(df, 14)

    fig_ADX = go.Figure()
    fig_ADX.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['ADX'],
            name='Average Directional Index'
        )
    )

    fig_ADX.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='Average Directional Index', template=temp_style
    )

    st.plotly_chart(fig_ADX, use_container_width=True)

    # aroon
    st.subheader('Aroon Indicator')
    st.write(
        "The Aroon indicator is a technical indicator that is used to identify trend changes in the price of an asset, as well as the strength of that trend. In essence, the indicator measures the time between highs and the time between lows over a time period.The indicator consists of the 'Aroon up' line, which measures the strength of the uptrend, and the 'Aroon down' line, which measures the strength of the downtrend.")

    aroon = ta.trend.AroonIndicator(df['Close'], 14)
    df['aroon_down'] = aroon.aroon_down()
    df['aroon_indicator'] = aroon.aroon_indicator()
    df['aroon_up'] = aroon.aroon_up()

    data_aroon = df.reset_index()
    fig_aroon = go.Figure()
    fig_aroon.add_trace(
        go.Scatter(
            x=data_aroon['Date'],
            y=data_aroon['aroon_down'],
            name='Aroon Down'
        )
    )

    fig_aroon.add_trace(
        go.Scatter(
            x=data_aroon['Date'],
            y=data_aroon['aroon_up'],
            name="Aroon up"
        )
    )

    fig_aroon.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='Aroon Indicator', template=temp_style
    )
    st.plotly_chart(fig_aroon, use_container_width=True)

    #plotting MACD
    st.subheader("Moving Average Convergance Divergence (MACD) ")
    st.write("Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a stock. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.")

    df_macd = calc_macd(df)
    df_macd = df_macd.reset_index()

    figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['Close'],
            name="Prices"
        )
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema12'],
            name='EMA12 '
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema26'],
            name='EMA26'

        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['macd'],
            name='MACD Line'
        ),
        row=2, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['signal'],
            name='Signal Line'
        ),
        row=2, col=1
    )

    figMACD.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0
    ),height=800,template=temp_style,title_text='Closing Price of Stock & MACD'

    )

    st.plotly_chart(figMACD, use_container_width=True)


    #plotting RSI
    st.subheader('Relative Strength Index (RSI)')
    st.write("The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.Traditionally the RSI is considered overbought when above 70 and oversold when below 30.")

    df_RSI = RSI(df,14)
    df_RSI=df_RSI.reset_index()

    fig_RSI = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)
    fig_RSI.add_trace(
        go.Scatter(
            x=df_RSI['Date'],
            y = df_RSI['Adj Close'],
            name='Closing Prices'
        ),
        row=1,col=1
    )

    fig_RSI.add_trace(
        go.Scatter(
            x=df_RSI['Date'],
            y=df_RSI['RSI'],
            name='RSI'
        ),
        row=2,col=1
    )

    fig_RSI.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0
    ),
        height=800, width=1000, title_text="Closing Price of Stock & RSI",template=temp_style)
    st.plotly_chart(fig_RSI, use_container_width=True)

    # TRIX
    st.subheader('TRIX Indicator')
    st.write(
        "The triple exponential average (TRIX) is a momentum indicator used by technical traders that shows the percentage change in a moving average that has been smoothed exponentially three times. "
        )

    trix = ta.trend.TRIXIndicator(df['Close'], 14)
    df['trix'] = trix.trix()

    data_trix = df.reset_index()
    fig_trix = go.Figure()
    fig_trix.add_trace(
        go.Scatter(
            x=data_trix['Date'],
            y=data_trix['trix'],
            name='TRIX'
        )
    )

    fig_trix.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='TRIX Indicator', template=temp_style
    )
    st.plotly_chart(fig_trix, use_container_width=True)

    #stc
    st.subheader('Schaff Trend Cycle (STC)')
    st.write("The Schaff trend cycle indicator is popular for a general trading strategy. The strategy suggests buying when it surges above 25 level and sell when the signal lines go below the 75 leve")

    stc = TA.STC(df, 14)

    fig_stc = go.Figure()
    fig_stc.add_trace(
        go.Scatter(
            x=df['Date'],
            y=stc,
            name="Schaff Trend Cycle"
        )
    )

    fig_stc.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='Schaff Trend Cycle', template=temp_style
    )

    st.plotly_chart(fig_stc, use_container_width=True)

    #Volume indicators
    st.header('Volume Indicators')
    st.write('Trading volume is a measure of how much a given financial asset has traded in a period of time. For stocks, volume is measured in the number of shares traded.Volume indicators are mathematical formulas that are visually represented in the most commonly used charting platforms.'
        )

    #OBV
    # plotting OBV
    st.subheader('On Balance Volume (OBV)')
    st.write(
        "On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price.")

    df['obv'] = OBV(df)

    fig_OBV = go.Figure()
    fig_OBV.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['obv'],
            name='On Balance Volume'
        )
    )

    fig_OBV.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='On Balance Volume', template=temp_style
    )

    st.plotly_chart(fig_OBV, use_container_width=True)

    #volatility indicators
    st.header('Volatility Indicators')
    st.write('The volatility indicator is a technical tool that measures how far security stretches away from its mean price, higher and lower. ')

    # plotting ATR
    st.subheader('Average True Range (ATR)')
    st.write(
        "Average True Range (ATR) is the average of true ranges over the specified period. ATR measures volatility, taking into account any gaps in the price movement.")

    df_ATR = ATR(df, 20)

    fig_ATR = go.Figure()
    fig_ATR.add_trace(
        go.Scatter(
            x=df_ATR['Date'],
            y=df_ATR['ATR'],
            name='Average True Range'
        )
    )

    fig_ATR.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0)
        , height=600, title_text='Average True Range', template=temp_style
    )

    st.plotly_chart(fig_ATR, use_container_width=True)


#short-term forecasting
if (feature == 'Next-Day Forecasting'):
    st.subheader('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

    # creating sidebar
    ticker = st.selectbox(
        'Enter or Choose NSE listed Stock Symbol',
        symbol, index=symbol.index('TRIDENT.NS'))

    def my_LSTM(ticker):
        try:
            start = dt.datetime.today() - dt.timedelta(5*365)
            end = dt.datetime.today()

            df = yf.download(ticker,start,end)
            df =  df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            st.write('It will take some seconds to fit the model....')
            data = df.sort_index(ascending=True, axis=0)
            new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
            for i in range(0, len(data)):
                new_data['Date'][i] = data['Date'][i]
                new_data['Close'][i] = data['Close'][i]

        # setting index
            new_data.index = new_data.Date
            new_data.drop('Date', axis=1, inplace=True)

        # creating train and test sets
            dataset = new_data.values

            train = dataset[0:987, :]
            valid = dataset[987:, :]

        # converting dataset into x_train and y_train
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            x_train, y_train = [], []
            for i in range(60, len(train)):
                x_train.append(scaled_data[i - 60:i, 0])
                y_train.append(scaled_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)

            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
            st.success('Model Fitted')
        # predicting 246 values, using past 60 from the train data
            inputs = new_data[len(new_data) - len(valid) - 60:].values
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)

            X_test = []
            for i in range(60, inputs.shape[0]):
                X_test.append(inputs[i - 60:i, 0])
            X_test = np.array(X_test)

            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            closing_price = model.predict(X_test)
            closing_price = scaler.inverse_transform(closing_price)


        # for plotting
            train = data[:987]
            valid = data[987:]
            valid['Predictions'] = closing_price

            st.write('#### Actual VS Predicted Prices')

            fig_preds = go.Figure()
            fig_preds.add_trace(
            go.Scatter(
                x=train['Date'],
                y=train['Adj Close'],
                name='Training data Closing price'
            )
            )

            fig_preds.add_trace(
            go.Scatter(
                x=valid['Date'],
                y=valid['Adj Close'],
                name='Validation data Closing price'
            )
            )

            fig_preds.add_trace(
            go.Scatter(
                x=valid['Date'],
                y=valid['Predictions'],
                name='Predicted Closing price'
            )
            )

            fig_preds.update_layout(legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='left',
            x=0)
            , height=600, title_text='Predictions on Validation Data', template='gridon'
            )

            st.plotly_chart(fig_preds, use_container_width=True)

        # metrics
            mae = mean_absolute_error(closing_price, valid['Adj Close'])
            rmse = np.sqrt(mean_squared_error(closing_price, valid['Adj Close']))
            accuracy = r2_score(closing_price, valid['Adj Close'])*100

            with st.container():
                st.write('#### Metrics')
                col_11, col_22, col_33 = st.columns(3)
                col_11.metric('Absolute error between predicted and actual value', round(mae,2))
                col_22.metric('Root mean squared error between predicted and actual value', round(rmse,2))

        # forecasting
            real_data = [inputs[len(inputs) - 60:len(inputs + 1), 0]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

            prediction = model.predict(real_data)
            prediction = scaler.inverse_transform(prediction)
            st.write('#### Next-Day Forecasting')

            with st.container():
                col_111, col_222, col_333 = st.columns(3)
                col_111.metric(f'Closing Price Prediction of the next trading day for {ticker} is',f' ₹ {str(round(float(prediction),2))}')

        except:
            st.warning("Oops! you can't go ahead!!")
            st.warning("The company you selected is listed newly...so we can't gather data.")

    my_LSTM(ticker)

if (feature == 'Pattern Recognition'):
    st.title('Pattern Recognition')
    st.write('A pattern is identified by a line that connects common price points, such as closing prices or highs or lows, during a specific period of time.')
    st.write('Technical analysts and chartists seek to identify patterns as a way to anticipate the future direction of a security’s price.')
    st.write('We automated this thing, for a specific ticker/symbol we scan through all the candlestick patterns and generate signals.')
    st.markdown('- Neutral - Not such activity or no trendline present at current moment')
    st.markdown('- Bullish - The stock is in up trendline ')
    st.markdown('- Bearish - The stock is in down trendline')
    st.write('#### Select Stock ')
    ticker_input = st.selectbox('Enter or Choose NSE listed stock', symbol)

    #plotiing prices

    show = st.radio(
        "Show/Hide Prices",
        ('Show', 'Hide'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    # closing price chart
    if (show == 'Show'):
        st.write("Enter period to check price of ", ticker_input)

        # getting date input
        min_value = dt.datetime.today() - dt.timedelta(10 * 365)
        max_value = dt.datetime.today()

        start_inputt = st.date_input(
            'Enter starting date',
            value=dt.datetime.today() - dt.timedelta(90),
            min_value=min_value, max_value=max_value, help='Enter the starting date from which you have to look the price'
        )

        end_inputt = st.date_input(
            'Enter last date',
            value=dt.datetime.today(),
            min_value=min_value, max_value=max_value, help='Enter the last date till which you have to look the price'
        )

        hist_price = yf.download(ticker_input, start_inputt, end_inputt)
        hist_price = hist_price.reset_index()


        # radio button ro switch between style
        chart = st.radio(
            "Choose Style",
            ('Candlestick', 'Line Chart'))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        if (chart == 'Line Chart'):
            # line chart plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_price['Date'],
                    y=hist_price['Adj Close'],
                    name='Closing price'
                )
            )

            fig.update_layout(
                title={
                    'text': 'Stock Prices of ' + ticker_input,
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}, height=600, template='gridon')
            fig.update_yaxes(tickprefix='₹')
            st.plotly_chart(fig, use_container_width=True)

        if (chart == 'Candlestick'):
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=hist_price['Date'],
                    open=hist_price['Open'],
                    high=hist_price['High'],
                    low=hist_price['Low'],
                    close=hist_price['Close'],
                    name='OHLC'
                )
            )

            fig.update_layout(
                title={
                    'text': 'Stock Prices of ' + ticker_input,
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}, height=600, template='gridon')
            fig.update_yaxes(tickprefix='₹')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write('Select show to check prices')

    # retrieving data from database
    start_input = dt.datetime.today() - dt.timedelta(365)
    end_input = dt.datetime.today()


    df = yf.download(ticker_input, start_input, end_input)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])



    #scanning
    candle_names = candlestick_patterns.keys()

    for candle,names in candlestick_patterns.items():
        df[candle] = getattr(talib, candle)(df['Open'], df['High'], df['Low'], df['Close'])
        last = df[candle].tail(1).values[0]


    tmp_df = df.drop(['Date', 'Open','High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1,)
    tmp_df_1 = tmp_df.T
    tmp_last = tmp_df_1.iloc[: , -1].tolist()
    signal_df = pd.DataFrame()
    signal_df['Pattern Names'] = candlestick_patterns.values()
    signal_df['Signal'] = tmp_last
    signal_df['Signal'] = signal_df['Signal'].map({0: 'Neutral', -100:'Bearish', 100:'Bullish'})

    #metrics
    bullish_count = len(signal_df[signal_df['Signal'] == 'Bullish'])
    bearish_count = len(signal_df[signal_df['Signal'] == 'Bearish'])

    with st.container():
        st.write('#### Overview of pattern recognition')
        coll_11, coll_22, coll_33 = st.columns(3)
        coll_11.metric('Patterns with bullish signals', bullish_count)
        coll_22.metric('Pattersn with bearish signals', bearish_count)

    @st.cache
    def color_survived(val):
        color = 'green' if val == 'Bullish' else 'red' if val == 'Bearish' else None
        return f'background-color: {color}'
    st.write('#### All candlestick patterns signals ')
    st.table(signal_df.style.applymap(color_survived, subset=['Signal']))

    sample = len(signal_df[signal_df['Signal'] == 'Bullish'])

if (feature == "Screener"):
    st.title('Screener')
    st.write("A stock screener is a set of tools that allow investors to quickly sort through various parameters of the stock")
    ticker_input = st.selectbox('Enter or Choose stock', symbol)
    start_input = dt.datetime.today() - dt.timedelta(120)
    end_input = dt.datetime.today()

    df = yf.download(ticker_input, start_input, end_input)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])

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
