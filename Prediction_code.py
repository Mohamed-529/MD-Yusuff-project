import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("apple stock/apple_stock.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Preprocess data for LSTM
def preprocess_data(df, features=['Close'], target_feature='Close', window_size=60):
    feature_data = df[features].values
    target_data = df[[target_feature]].values

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    feature_data_scaled = feature_scaler.fit_transform(feature_data)
    target_data_scaled = target_scaler.fit_transform(target_data)

    X, y = [], []
    for i in range(window_size, len(feature_data_scaled)):
        X.append(feature_data_scaled[i-window_size:i])
        y.append(target_data_scaled[i, 0])  # predicting only 'Close'

    return np.array(X), np.array(y), feature_scaler, target_scaler

df=load_data()
x, y, feature_scaler, target_scaler = preprocess_data(df, features=['Open','High','Low','Close','Volume'], target_feature='Close')


# Build simple LSTM
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Streamlit UI
st.title("Apple Stock Price Prediction")
df = load_data()


st.write("Stockdata loaded for Apple single stock dataset")
st.line_chart(df['Close'])

window_size = 60
features=['Open','High','Low','Close','Volume']
X, y, feature_scaler, target_scaler = preprocess_data(df,features=features)
X = X.reshape((X.shape[0], X.shape[1], len(features)))

if st.button("Train Model and Predict"):
    with st.spinner("Training model..."):
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        last_sequence = X[-1].reshape((1, window_size, X.shape[2]))
        future_preds = []

        for _ in range(30):
            pred = model.predict(last_sequence)[0][0]
            future_preds.append(pred)
            # Only replacing the 'Close' value (1st feature) for prediction
            new_entry = last_sequence[:, 1:, :].copy()
            new_feature_vector=[0,0,0,pred,0]
            new_entry = np.append(new_entry,[[new_feature_vector]], axis=1)
            last_sequence=new_entry

        # Inverse scale the predictions to get actual prices
        predicted_prices = target_scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Generate future dates
        last_date = pd.to_datetime(df.index[-1])
        n_steps = len(predicted_prices)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq='B')

# Create DataFrame for predictions
        pred_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Close'])



    st.success("✅Prediction completed successfully!")

    st.write("Future Predictions")
    st.line_chart(pred_df)

    fig, ax = plt.subplots()
    df['Close'].plot(ax=ax, label='Historical')
    pred_df['Predicted Close'].plot(ax=ax, label='Predicted')
    ax.legend()
    st.pyplot(fig)
