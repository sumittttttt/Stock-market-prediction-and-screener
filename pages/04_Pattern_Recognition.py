import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
from functions import *
import plotly.graph_objects as go
from patterns import candlestick_patterns
import talib


csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(0, len(symbol)):
    symbol[i] = symbol[i] + ".NS"


st.title('Pattern Recognition')
st.write('A pattern is identified by a line that connects common price points, such as closing prices or highs or lows, during a specific period of time.')
st.write('Technical analysts and chartists seek to identify patterns as a way to anticipate the future direction of a security’s price.')
st.write('We automated this thing, for a specific ticker/symbol we scan through all the candlestick patterns and generate signals.')
st.markdown('- Neutral - Not such activity or no trendline present at current moment')
st.markdown('- Bullish - The stock is in up trendline ')
st.markdown('- Bearish - The stock is in down trendline')
st.write('#### Select Stock ')
ticker_input = st.selectbox('Enter or Choose NSE listed stock', symbol,index=symbol.index('VISHWARAJ.NS'))

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

        start_input = st.date_input(
            'Enter starting date',
            value=dt.datetime.today() - dt.timedelta(90),
            min_value=min_value, max_value=max_value, help='Enter the starting date from which you have to look the price'
        )

        end_input = st.date_input(
            'Enter last date',
            value=dt.datetime.today(),
            min_value=min_value, max_value=max_value, help='Enter the last date till which you have to look the price'
        )

        hist_price = yf.download(ticker_input, start_input, end_input)
        hist_price = hist_price.reset_index()
        hist_price['Date'] = pd.to_datetime(hist_price['Date']).dt.date



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
df['Date'] = pd.to_datetime(df['Date']).dt.date


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
