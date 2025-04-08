import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd
import dearpygui.dearpygui as dpg

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
                         (df[('Close', ticker)] >= df[(moving_average, '')]))
    df['Cross_Below'] = ((df['Previous_Close'] > df[moving_average]) &
                         (df[('Close', ticker)] <= df[(moving_average, '')]))

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
        plt.scatter(buy_signals.index, buy_signals['Close'], color='green',
                    label='Buy Signal', marker='o', s=50)

        sell_signals = df[df['Final_Sell']]
        plt.scatter(sell_signals.index, sell_signals['Close'], color='red',
                    label='Sell Signal', marker='o', s=50)

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

    x = []

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

    for i in range(len(df)):
        x.append(i)

    dpg.create_context()

    with dpg.window(label="Tutorial", height=800, width=1400, no_close=True, no_move=True, no_resize=True):
        with dpg.plot(label="Line Series", height=800, width=1400):
            dpg.add_plot_axis(dpg.mvXAxis, label="x")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="y_axis")

            dpg.add_line_series(x, ema, label="Exponential Moving Average", parent="y_axis")
            dpg.add_candle_series(dates=x,
                                  opens=open_prices, closes=close_prices,
                                  lows=low_prices, highs=high_prices,
                                  label="CANDLE", parent='y_axis')

            dpg.set_axis_limits(y_axis, min(low_prices) - 10, max(high_prices) + 10)

    dpg.create_viewport(title='Custom Title', width=1400, height=800, resizable=False)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


while True:
    user_input = input("""Enter valid ticker
>>> """).upper()
    graph(user_input.replace(" CHART", ""))
