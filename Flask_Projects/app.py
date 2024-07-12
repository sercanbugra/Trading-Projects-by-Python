# app.py

from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def stock_chart():
    chart = None
    ticker = None
    period = None

    if request.method == 'POST':
        ticker = request.form['ticker']
        period = request.form['period']

        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        # Create a simple plot (you can modify this based on your actual plot requirements)
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Close'], marker='o', linestyle='-')
        plt.title(f'Stock Chart for {ticker} ({period})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Encode image to base64 string
        buf.seek(0)
        chart = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('stock_chart.html', chart=chart, ticker=ticker, period=period)

if __name__ == '__main__':
    app.run(debug=True)
