import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Create list of cryptocurrencies
cryptos = ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','STETH-USD','DOGE-USD','HEX-USD','SOL-USD','MATIC-USD','WTRX-USD',
           'TRX-USD','DOT-USD','LTC-USD','BUSD-USD','SHIB-USD','AVAX-USD','DAI-USD','WBTC-USD','LINK-USD','LEO-USD','ATOM-USD','UNI7083-USD','XMR-USD',
           'OKB-USD','ETC-USD','XLM-USD','TON11419-USD']

# Create sidebar for inputs
st.sidebar.header('Crypto Price Tracker: Visualizing Price Data')
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
plot_type = st.sidebar.selectbox('Plot Type', ('Single Coin', 'Comparison'))
crypto_1 = st.sidebar.selectbox('Select cryptocurrency 1', cryptos)
crypto_2 = st.sidebar.selectbox('Select cryptocurrency 2', cryptos)

# Create function to get cryptocurrency price data
def get_crypto_price_data(crypto, start_date, end_date):
    data = yf.download(crypto, start=start_date, end=end_date)
    return data

# Create function to plot cryptocurrency price data
def plot_crypto_price_data(crypto, start_date, end_date):
    price_data = get_crypto_price_data(crypto, start_date, end_date)
    plt.plot(price_data['Adj Close'])
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{crypto} Price Data')
    plt.xticks(rotation=45)

# Create function to plot comparison cryptocurrency price data
def plot_comparison_crypto_price_data(crypto_1, crypto_2, start_date, end_date):
    price_data_1 = get_crypto_price_data(crypto_1, start_date, end_date)
    price_data_2 = get_crypto_price_data(crypto_2, start_date, end_date)
    plt.plot(price_data_1['Adj Close'], label=crypto_1)
    plt.plot(price_data_2['Adj Close'], label=crypto_2)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{crypto_1} vs {crypto_2} Price Data')
    plt.xticks(rotation=45)
    plt.legend()

# Determine the type of plot to display
if plot_type == 'Single Coin':
    plot_crypto_price_data(crypto_1, start_date, end_date)
elif plot_type == 'Comparison':
    plot_comparison_crypto_price_data(crypto_1, crypto_2, start_date, end_date)

# Display plot
st.pyplot(plt)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# retrieve historical data from Yahoo Finance API
def get_historical_data(ticker_symbol, n_years):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=n_years)
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    return df

# prepare data
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaler, scaled_data

# define function for creating dataset
def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:(i+time_steps), 0])
        Y.append(dataset[i+time_steps, 0])
    return np.array(X), np.array(Y)

# train model
def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train)

# make predictions
def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    return predictions

# Streamlit code
st.title('Cryptocurrency Prediction')

ticker_symbol = st.selectbox(
    'Select cryptocurrency',
    ['BTC-USD','ETH-USD','USDT-USD','BNB-USD','USDC-USD','XRP-USD','ADA-USD','STETH-USD','DOGE-USD','HEX-USD','SOL-USD','MATIC-USD','WTRX-USD',
    'TRX-USD','DOT-USD','LTC-USD','BUSD-USD','SHIB-USD','AVAX-USD','DAI-USD','WBTC-USD','LINK-USD','LEO-USD','ATOM-USD','UNI7083-USD','XMR-USD',
    'OKB-USD','ETC-USD','XLM-USD','TON11419-USD']
)
n_years = st.slider('Number of years to retrieve', 1, 10, 5)
n_days = st.slider('Number of days to predict', 1, 365, 30)

if st.button('Update Plot'):
    # retrieve historical data
    df = get_historical_data(ticker_symbol, n_years)

    # prepare data
    scaler, scaled_data = prepare_data(df)

    # split data into train and test sets
    train_data = scaled_data[:int(len(df)*0.8)]
    test_data = scaled_data[int(len(df)*0.8):]

    # create dataset for training
    time_steps = 10
    X_train, Y_train = create_dataset(train_data, time_steps)
    X_test, Y_test = create_dataset(test_data, time_steps)

    # create and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_model(model, X_train, Y_train.ravel())

    # make predictions
    predictions = make_predictions(model, X_test, scaler)

    # invert actual prices to their original scale
    Y_test = scaler.inverse_transform([Y_test.ravel()])[0]

    # calculate model accuracy
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    # normalize the values
    mae_percent = mae / (Y_test.max() - Y_test.min()) * 100
    r2_percent = (r2 + 1) / 2 * 100  # R2 score can be negative

    st.write("Model Accuracy:")
    st.write(f"MAE: {mae_percent:.2f}%")
    st.write(f"R2 Score: {r2_percent:.2f}%")

    # plot the predictions
    fig, ax = plt.subplots()
    ax.plot(predictions, label='Predictions')
    ax.plot(Y_test, label='Actual values')
    ax.legend()
    st.pyplot(fig)

    # predict price for the next n_days
    last_n_days = scaled_data[-time_steps:]
    last_n_days = np.reshape(last_n_days, (1, time_steps))

    predicted_price = model.predict(last_n_days)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    st.write(f"Predicted price for the next {n_days} days: {predicted_price[0][0]:.2f}")

   


