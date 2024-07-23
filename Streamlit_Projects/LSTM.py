import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error

start = '2023-01-01'
end = '2024-07-23'
stock = 'MSFT'

data = yf.download(stock, start, end)

data.reset_index(inplace=True)

ma_100_days = data.Close.rolling(100).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100'))
fig1.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close Price'))
fig1.update_layout(title='Price vs MA100', xaxis_title='Date', yaxis_title='Price')
fig1.show()

ma_200_days = data.Close.rolling(200).mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
fig2.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close Price', line=dict(color='green')))
fig2.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
fig2.show()

data.dropna(inplace=True)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

data_train_scale = scaler.fit_transform(data_train)

x = []
y = []

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])

x, y = np.array(x), np.array(y)

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation='relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units =1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x,y, epochs = 10, batch_size =32, verbose =1)

pas_100_days = data_train.tail(100)

data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

data_test_scale  =  scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

y_predict = model.predict(x)

scale =1/scaler.scale_

y_predict = y_predict*scale
y = y*scale

mae = mean_absolute_error(y, y_predict)
mse = mean_squared_error(y, y_predict)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data_test.index, y=y_predict.flatten(), mode='lines', name='Predicted Price'))
fig3.add_trace(go.Scatter(x=data_test.index, y=y.flatten(), mode='lines', name='Original Price'))
fig3.update_layout(title='Original Price vs Predicted Price', xaxis_title='Date', yaxis_title='Price')
fig3.show()

model.save('LSTM.h5')