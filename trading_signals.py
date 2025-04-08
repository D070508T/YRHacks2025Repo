import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np

def create_trading_chart():
    try:
        print("Downloading AAPL data...")
        df = yf.download("AAPL", period="1mo", interval="1d")
        
        if df.empty:
            raise ValueError("No data returned from yfinance")
            
        # Convert data ensuring proper types
        dates = df.index.to_numpy(dtype='datetime64[s]').flatten()
        opens = df['Open'].to_numpy(dtype=np.float64).flatten()
        highs = df['High'].to_numpy(dtype=np.float64).flatten()
        lows = df['Low'].to_numpy(dtype=np.float64).flatten()
        closes = df['Close'].to_numpy(dtype=np.float64).flatten()
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            increasing_line_color='#2ECC71',
            decreasing_line_color='#E74C3C'
        )])

        # Add buy/sell signals (example - replace with your actual signals)
        buy_signals = [False, True, False, False, True] + [False]*(len(dates)-5)
        sell_signals = [False, False, True, False, False] + [False]*(len(dates)-5)
        
        if any(buy_signals):
            fig.add_trace(go.Scatter(
                x=dates[buy_signals],
                y=closes[buy_signals],
                mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                name='Buy Signal'
            ))
        
        if any(sell_signals):
            fig.add_trace(go.Scatter(
                x=dates[sell_signals],
                y=closes[sell_signals],
                mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down'),
                name='Sell Signal'
            ))

        fig.update_layout(
            title='AAPL with Trading Signals',
            xaxis_rangeslider_visible=False
        )
        
        fig.write_html("trading_signals.html", auto_open=True)
        print("Trading chart generated successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Fallback to sample data if needed
        create_sample_chart()

def create_sample_chart():
    # Sample data implementation here
    pass

if __name__ == "__main__":
    create_trading_chart()
