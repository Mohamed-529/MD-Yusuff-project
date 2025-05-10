import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load data
df = pd.read_csv('stock_data.csv')

# Preprocess data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Split data
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[0:train_size], df_scaled[train_size:]

# Create LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(train_data, epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(test_data)
