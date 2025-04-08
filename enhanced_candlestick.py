import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np

def create_candlestick_chart():
    try:
        print("Downloading AAPL data...")
        df = yf.download("AAPL", period="1mo", interval="1d")
        
        if df.empty:
            raise ValueError("No data returned from yfinance")
            
        # Convert data ensuring proper types
        dates = df.index.to_numpy(dtype='datetime64[s]')  # Proper datetime conversion
        opens = df['Open'].to_numpy(dtype=np.float64).flatten()     # Convert to 1D array
        highs = df['High'].to_numpy(dtype=np.float64).flatten()
        lows = df['Low'].to_numpy(dtype=np.float64).flatten()
        closes = df['Close'].to_numpy(dtype=np.float64).flatten()
        
        # Comprehensive debug prints
        print("\n=== DATA VALIDATION ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"Dates array shape: {dates.shape}, type: {dates.dtype}")
        print(f"Opens array shape: {opens.shape}, type: {opens.dtype}")
        print("\nFirst 3 data points:")
        for i in range(3):
            print(f"Date: {dates[i]}, Open: {opens[i]}, High: {highs[i]}, Low: {lows[i]}, Close: {closes[i]}")
        
        print("\nLast 3 data points:")
        for i in range(-3, 0):
            print(f"Date: {dates[i]}, Open: {opens[i]}, High: {highs[i]}, Low: {lows[i]}, Close: {closes[i]}")
        
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
        
        fig.update_layout(
            title='AAPL Stock: Last 30 Days',
            xaxis_rangeslider_visible=False
        )
        
        fig.write_html("enhanced_candlestick.html", auto_open=True)
        print("Chart generated successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Fallback to sample data if needed
        create_sample_chart()

def create_sample_chart():
    # Sample data implementation here
    pass

if __name__ == "__main__":
    create_candlestick_chart()
