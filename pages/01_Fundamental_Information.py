import streamlit as st
import yfinance as yf
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title('Fundamental Information')

csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(0, len(symbol)):
    symbol[i] = symbol[i] + ".NS"

ticker = st.selectbox(
        'Enter or Choose NSE listed Stock Symbol',
        symbol)
stock = yf.Ticker(ticker)

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

#getting data

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


hist_price = yf.download(ticker,start_input,end_input)
hist_price = hist_price.reset_index()
hist_price['Date'] = pd.to_datetime(hist_price['Date']).dt.date

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

historical_csv = convert_df(hist_price)
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
            x=hist_price['Date'],
            y=hist_price['Adj Close'],
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
            x=hist_price['Date'],
            open=hist_price['Open'],
            high=hist_price['High'],
            low=hist_price['Low'],
            close=hist_price['Close'],
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
st.dataframe(quarterly_results.style.highlight_max(axis=1, color='lightgreen'))


#profit and loss
st.subheader('Profit & Loss')
st.write("A profit and loss (P&L) statement is a annually financial report that provides a summary of a company's revenue, expenses and profit.")
financials = stock.financials
financials.columns = financials.columns.date
financials.dropna(axis=0, inplace=True)
financials = financials.astype('int64')
for i in financials.columns:
    financials[i] = financials.apply(lambda x: "{:,}".format(x[i]), axis=1)
st.dataframe(financials.style.highlight_max(axis=1,color='lightgreen'))

#balance sheet
st.subheader('Balance Sheet')
st.write("A balance sheet is a financial statement that reports a company's assets, liabilities, and shareholder equity.")
balance = stock.balance_sheet
balance.columns = balance.columns.date
balance.dropna(axis=0, inplace=True)
balance = balance.astype('int64')
for i in balance.columns:
    balance[i] = balance.apply(lambda x: "{:,}".format(x[i]), axis=1)
st.dataframe(balance.style.highlight_max(axis=1,color='lightgreen'))

#cash flow
st.subheader('Cash Flows')
st.write("The term cash flow refers to the net amount of cash and cash equivalents being transferred in and out of a company.")
cf = stock.cashflow
cf.columns = cf.columns.date
cf.dropna(axis=0, inplace=True)
cf = cf.astype('int64')
for i in cf.columns:
    cf[i] = cf.apply(lambda x: "{:,}".format(x[i]), axis=1)
st.dataframe(cf.style.highlight_max(axis=1,color='lightgreen'))

#actions
st.subheader('Splits & Dividends')
st.write('')
actions = stock.actions
actions.index = actions.index.date
st.dataframe(actions, width=1000)
