import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)  # None means no limit

frequencies = {
    '2m': '2min',
    '15m': '15min',
    '1h': '1h',
    '90m': '90min',
    '1d': '1D',
    '5d': '5D',
    '1wk': '1W',
    '1mo': '1ME',
    '3mo': '3ME'
}

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Allow the output to be as wide as needed
pd.set_option('display.max_colwidth', None)  # To prevent truncation of column content


# Download stock data
def get_stock_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    print(df)
    df.index = pd.date_range(start=df.index[0], periods=len(df), freq=frequencies.get(interval))
    return df


# Compute moving averages (SMA & EMA)
def compute_moving_averages(df, window=25):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['EMA'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df


# Compute Relative Strength Index
def relative_strength_index(df, strength=7):
    df['Price_Change'] = df['Close'].diff()
    df['Gain'] = df['Price_Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Price_Change'].apply(lambda x: -x if x < 0 else 0)

    # Use EMA for smoothing (Wilder's method)
    df['Avg_Gain'] = df['Gain'].ewm(alpha=1 / strength, adjust=False).mean()
    df['Avg_Loss'] = df['Loss'].ewm(alpha=1 / strength, adjust=False).mean()

    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))

    return df


# Computes RSI signals
def rsi_signals(df, low=36, high=68, lookback_period=5):
    df['RSI_Buy_Signal'] = (df['RSI'].shift(1) < low) & (df['RSI'] >= low).astype(bool)
    df['RSI_Sell_Signal'] = (df['RSI'].shift(1) > high) & (df['RSI'] <= high).astype(bool)

    # Check if a signal happened in the last `lookback_period` days
    df['Recent_RSI_Buy'] = df['RSI_Buy_Signal'].rolling(lookback_period).max().astype(bool)
    df['Recent_RSI_Sell'] = df['RSI_Sell_Signal'].rolling(lookback_period).max().astype(bool)
    return df


def detect_ma_crossovers(df, ticker, lookback_period=5, moving_average='EMA'):
    # Create the 'Previous_Close' column
    df['Previous_Close'] = df['Close'].shift(1).fillna(0)

    df['Cross_Above'] = ((df['Previous_Close'] < df[moving_average]) &
                         (df[('Close', ticker)] >= df[moving_average]))
    df['Cross_Below'] = ((df['Previous_Close'] > df[moving_average]) &
                         (df[('Close', ticker)] <= df[moving_average]))

    # Check if a crossover happened in the last `lookback_period` days
    df['Recent_Cross_Above'] = df['Cross_Above'].rolling(lookback_period).max().astype(bool)
    df['Recent_Cross_Below'] = df['Cross_Below'].rolling(lookback_period).max().astype(bool)

    return df


def generate_signals(df):
    df['Full_Buy'] = df['Cross_Above'] & df['Recent_RSI_Buy']
    df['Full_Sell'] = df['Cross_Below'] & df['Recent_RSI_Sell']

    # Conflict Handling: If both Buy & Sell are true, cancel out
    df['Final_Buy'] = (df['Full_Buy']) & (~df['Full_Sell'])
    df['Final_Sell'] = (df['Full_Sell']) & (~df['Full_Buy'])

    return df

# Main function
def graph(company_ticker):
    ticker = company_ticker.upper()
    period = input("""
Enter a valid timeframe
[1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
>>> """)

    if period == '1d':
        df = get_stock_data(ticker, '1d', '2m')
    elif period == '5d':
        df = get_stock_data(ticker, '5d', '5m')
    elif period == '1mo':
        df = get_stock_data(ticker, '1mo', '30m')
    elif period == '3mo':
        df = get_stock_data(ticker, '3mo', '60m')
    elif period == '1y':
        df = get_stock_data(ticker, '1y', '1d')
    elif period == '2y':
        df = get_stock_data(ticker, '2y', '1d')
    elif period == '5y':
        df = get_stock_data(ticker, '5y', '1wk')
    else:  # assume 10y
        df = get_stock_data(ticker, '10y', '1wk')

    length = len(df)

    moving_average_strength = 25
    RSI_strength = 7
    RSI_low_threshold = 36
    RSI_high_threshold = 68
    lookback_period = 5

    df = compute_moving_averages(df, moving_average_strength)
    df = relative_strength_index(df, RSI_strength)
    df = rsi_signals(df, RSI_low_threshold, RSI_high_threshold, lookback_period)
    df = detect_ma_crossovers(df, company_ticker, lookback_period, 'EMA')
    df = generate_signals(df)
    df = df.iloc[:length]

    open_prices = df['Open'].astype(float).values
    close_prices = df['Close'].astype(float).values
    high_prices = df['High'].astype(float).values
    low_prices = df['Low'].astype(float).values
    ema = df['EMA'].astype(float).values

    dates = df.index.to_numpy(dtype='datetime64[s]')
    opens = df['Open'].to_numpy(dtype=np.float64).flatten()
    highs = df['High'].to_numpy(dtype=np.float64).flatten()
    lows = df['Low'].to_numpy(dtype=np.float64).flatten()
    closes = df['Close'].to_numpy(dtype=np.float64).flatten()
    
    print("\nTuple data validation:")
    print(f"First date: {dates[0]}")
    print(f"First open/high/low/close: {opens[0]}, {highs[0]}, {lows[0]}, {closes[0]}")
    
    # Create candlestick chart with verified data
    fig = go.Figure(data=[go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        increasing_line_color='#2ECC71',  # Green
        decreasing_line_color='#E74C3C',  # Red
        increasing_fillcolor='#2ECC71',
        decreasing_fillcolor='#E74C3C'
    )])

    # Add clean buy/sell signal visualization
    if 'Final_Buy' in df.columns:
        buy_signals = df[df['Final_Buy']]
        y_coords = buy_signals['Close'].to_numpy(dtype=np.float64).flatten()
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    color='#00FF00',  # Bright green
                    size=14,
                    symbol='triangle-up',
                    line=dict(width=1, color='black')
                ),
                name='BUY',
                text='BUY',
                textposition='top center'
            ))
    
    if 'Final_Sell' in df.columns:
        sell_signals = df[df['Final_Sell']]
        y_coords = sell_signals['Close'].to_numpy(dtype=np.float64).flatten()
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=y_coords,
                mode='markers+text',
                marker=dict(
                    color='#FF0000',  # Bright red
                    size=14,
                    symbol='triangle-down',
                    line=dict(width=1, color='black')
                ),
                name='SELL',
                text='SELL',
                textposition='bottom center'
            ))

    # Dynamic axis scaling
    price_range = df['High'].max() - df['Low'].min()
    padding = price_range * 0.05
    
    fig.update_layout(
        title=f'{ticker} Stock with Trading Signals',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis_range=[df['Low'].min()-padding, df['High'].max()+padding],
        xaxis_rangeslider_visible=True
    )
    
    output_file = "enhanced_candlestick.html"
    fig.write_html(output_file, auto_open=True)

while True:
    user_input = input("""Enter valid ticker
>>> """).upper()
    graph(user_input.replace(" CHART", ""))
