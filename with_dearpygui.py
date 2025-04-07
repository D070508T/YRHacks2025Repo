import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd
import dearpygui.dearpygui as dpg


frequencies = {
    '2m': '2min',
    '15m': '15min',
    '1h': '1h',
    '90m': '90min',
    '1d': '1D',
    '5d': '5D',
    '1wk': '1W',
    '1mo': '1M',
    '3mo': '3M'
}

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Allow the output to be as wide as needed
pd.set_option('display.max_colwidth', None)  # To prevent truncation of column content


# Download stock data
def get_stock_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
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

    df['Cross_Above'] = (df['Previous_Close'] < df[moving_average]) & (df[('Close', ticker)] >= df[(moving_average, '')])
    df['Cross_Below'] = (df['Previous_Close'] > df[moving_average]) & (df[('Close', ticker)] <= df[(moving_average, '')])

    # Check if a crossover happened in the last `lookback_period` days
    df['Recent_Cross_Above'] = df['Cross_Above'].rolling(lookback_period).max().astype(bool)
    df['Recent_Cross_Below'] = df['Cross_Below'].rolling(lookback_period).max().astype(bool)

    return df


def generate_signals(df):
    if input("""
Which signal should be searched for first?
Relative strength index [RSI] (default option)
or moving average [MA]
>>> """).upper() == "MA":
        df['Full_Buy'] = df['RSI_Buy_Signal'] & df['Recent_Cross_Above']
        df['Full_Sell'] = df['RSI_Sell_Signal'] & df['Recent_Cross_Below']
    else:
        df['Full_Buy'] = df['Cross_Above'] & df['Recent_RSI_Buy']
        df['Full_Sell'] = df['Cross_Below'] & df['Recent_RSI_Sell']

    # Conflict Handling: If both Buy & Sell are true, cancel out
    df['Final_Buy'] = (df['Full_Buy']) & (~df['Full_Sell'])
    df['Final_Sell'] = (df['Full_Sell']) & (~df['Full_Buy'])

    return df


# Plot stock trends
def plot_stock_trends(df, ticker, period, interval, low=36, high=68, moving_average='EMA'):
    plt.figure(figsize=(12, 6))
    chartType = input("""
To display RSI, hit 'RSI'
To display RSI and closing, hit 'BOTH'
To display closing chart, hit [ENTER]
>>> """).upper()
    title = f'{ticker} {period} Period {interval} Interval Closing Price + Relative Strength Index'
    y = 'Price + RSI'
    if chartType == 'RSI' or chartType == 'BOTH':
        plt.plot(df['RSI'], label='RSI', color='grey')
        plt.axhline(low, color='blue', linestyle='--')
        plt.axhline(high, color='blue', linestyle='--')
        title = f'{ticker} {period} Relative Strength Index'
        y = 'RSI'
    if chartType != 'RSI':
        plt.plot(df['Close'], label='Closing Price', color='black')
        plt.plot(df['SMA'], label='SMA', color='blue')
        plt.plot(df['EMA'], label='EMA', color='red')
        title = f'{ticker} {period} Period {interval} Interval Closing Price'
        y = 'Price'

        buy_signals = df[df['Final_Buy']]
        plt.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Buy Signal', marker='o', s=50)

        sell_signals = df[df['Final_Sell']]
        plt.scatter(sell_signals.index, sell_signals['Close'], color='red', label='Sell Signal', marker='o', s=50)

    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(y)
    plt.legend()

    print(df[[
        'Close', 'Previous_Close',
        moving_average, 'RSI',
        'Cross_Above', 'Cross_Below',
        'Recent_Cross_Above', 'Recent_Cross_Below',
        'RSI_Buy_Signal', 'RSI_Sell_Signal',
        'Final_Buy', 'Final_Sell'
    ]])

    plt.show()


# Main function
def graph(company_ticker):
    ticker = company_ticker.upper()
    period = input("""
Enter a valid timeframe
[1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
>>> """)

    if period.endswith('d'):
        number = int(period[:-1])
        df = get_stock_data(ticker, period, '2m')
        for i in range(len(df)-1, 0, -1):
            if i % number != 0:
                df = df.drop(df.index[i])
    elif period.endswith('w'):
        number = int(period[:-1])
        period = str((number*7)) + 'd'
        df = get_stock_data(ticker, period, '15m')
        for i in range(len(df)-1, 0, -1):
            if i % number != 0:
                df = df.drop(df.index[i])
    elif period.endswith('mo'):
        number = int(period[:-2])
        df = get_stock_data(ticker, period, '1h')
        for i in range(len(df)-1, 0, -1):
            if i % number != 0:
                df = df.drop(df.index[i])
    elif period.endswith('y'):
        number = int(period[:-1])
        df = get_stock_data(ticker, period, '1d')
        for i in range(len(df)-1, 0, -2):
            df = df.drop(df.index[i])

        for i in range(len(df)-1, 0, -1):
            if i % number != 0:
                df = df.drop(df.index[i])

    length = len(df)

    df = compute_moving_averages(df, 25)
    df = relative_strength_index(df, 7)
    df = rsi_signals(df, 36, 68, 5)
    df = detect_ma_crossovers(df, company_ticker, 5, 'EMA')
    df = generate_signals(df)
    df = df.iloc[:length]

    x = []
    open = df['Open'].astype(float).values
    close = df['Close'].astype(float).values
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values
    ema = df['EMA'].astype(float).values

    for i in range(len(df)):
        x.append(i)

    dpg.create_context()

    with dpg.window(label="Tutorial", height=800, width=800, no_close=True, no_move=True, no_resize=True):
        with dpg.plot(label="Line Series", height=800, width=800):
            dpg.add_plot_axis(dpg.mvXAxis, label="x")
            dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")
            dpg.add_line_series(x, ema, label="Exponential Moving Average", parent="y_axis")
            dpg.add_candle_series(dates=x, opens=open, closes=close, lows=low, highs=high, label="CANDLE", parent='y_axis')

    dpg.create_viewport(title='Custom Title', width=820, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

while True:
    user_input = input("""Enter valid ticker
>>> """).upper()
    graph(user_input.replace(" CHART", ""))