# Stock market prediction and screener 📈

### Predict Stocks is the all in one financial website for the retail investors where retail investors can take a look at all Fundamental Information, Technical Indicators, Screeners, Pattern Recognition and Next-Day Forecasting of all the National Stock Exchange (NSE) listed stocks. 


## 1. Features of the Machine Learning based Web-app 
1. One can gain information of all the stocks that are listed on National Stock Exchange (NSE) (To Be Exact - 1773 Companies).
2. Webapp have features like Fundamental Information, Technical Indicators, Screener, Pattern Recognition, Next-Day Forecasting (Machine Learning based).

## 2. Details related to stock prices 
1. All the historical prices for last 10 years are taken from Yahoo Finance.
2. Time period for stock prices is one day.

## 3. Fundamental Information feature overview 
1. It contains all the information related to company.
2. Historical prices of the company of last 10 years with candlestick and line chart.
3. One can download historical prices too.
4. All quarterly and annually Financial Results, Balance Sheet, Cash Flow and Splits & Dividends


## 4. Technical Indicators feature overview 
1. It contains all the well known indicators that traders and investors use while investing.
2. More than 11 indicators are there.

## 5. Technical Screener feature overview
1. It contains all the important parameters or metrics that are related to company.
2. It also contains one more important feature, that is it give signals whether selected stock is breaking out or not. Traders do this manually by looking at candles.
3. With breaking out it also gives signals for consolidating or not.

## 6. Pattern Recognition feature overview
1. It is important feature of the webapp.
2. Traders do this manually by looking at candles.
3. We automate this thing by scanning all the candlestick patterns for the selected stock, then it generates signals whether it is bullish or bearish


## 7. Next-Day Forecasting feature overview
1. We have build the efficient Machine Learning model to predict the next day price.
2. Our model trained on past 5 years of historical data and while predicting it looks for past 2 months ton predict next-day price.



## How to run on the localhost?
1. Install all the required libraries from the requirements.txt
2. Then open the code in your IDE and just run this command in terminal
```
streamlit run webapp.py
```

## Why not deployed anywhere?
Because having a problem with TA-Lib installation.

For the mean time here is the video
