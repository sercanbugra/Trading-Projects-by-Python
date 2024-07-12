from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
from django.http import HttpResponse
import yfinance as yf
import mplfinance as mpf
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from .forms import StockForm

def stock_chart(request):
    chart = None
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            period = form.cleaned_data['period']
            
            # Fetch stock data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            # Create a candlestick chart
            fig, ax = plt.subplots()
            mpf.plot(df, type='candle', style='charles', ax=ax)
            
            # Convert plot to image
            buf = io.BytesIO()
            canvas = FigureCanvas(fig)
            canvas.print_png(buf)
            plt.close(fig)

            # Encode image to base64 string
            chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        form = StockForm()

    return render(request, 'stocks/stock_chart.html', {'form': form, 'chart': chart})