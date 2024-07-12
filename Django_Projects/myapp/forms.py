from django import forms

class StockForm(forms.Form):
    TICKER_CHOICES = [
        ('AAPL', 'Apple'),
        ('GOOGL', 'Google'),
        ('MSFT', 'Microsoft'),
        ('AMZN', 'Amazon'),
    ]
    PERIOD_CHOICES = [
        ('1wk', '1 Week'),
        ('1mo', '1 Month'),
        ('6mo', '6 Months'),
    ]

    ticker = forms.ChoiceField(choices=TICKER_CHOICES, label='Stock Ticker')
    period = forms.ChoiceField(choices=PERIOD_CHOICES, label='Period')