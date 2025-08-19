import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

st.title("Time Series Forecasting for a stock Using XGBoost with Train - Test Split and Lag Features")

with st.sidebar.form("Details of Stock"):
    st.write("Enter the Details:")
    stock = st.text_input("Enter the stock name: (eg AAPL)", value="AAPL")
    start_date = st.date_input("Enter the Start Date: ", value="2020-01-01")
    end_date = st.date_input("Enter the End Date: ", value = "2021-01-01")

    submitted = st.form_submit_button("Submit")
    
if submitted:
    ticker = [stock]

    try: 
        df = yf.download(tickers = ticker, start = start_date, end = end_date)
    except Exception as e:
        st.write("Failed to Download. Please try again in some time.")
    
    data = df.copy()
    data['Close_lag1'] = data['Close'].shift(1)
    data['Volume_lag1'] = data['Volume'].shift(1)
    data = data.dropna()

    plt.figure()
    plt.plot(data["Close"], label = f"{ticker[0]}")
    plt.title(f"{ticker[0]} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt.gcf())

    train_data = data.iloc[:int(len(data)*0.85+1)]
    test_data = data.iloc[int(len(data)*0.85):] 

    plt.figure()
    plt.plot(train_data["Close"], label = "Training Data")
    plt.plot(test_data["Close"], label = "Test Data")
    plt.title("Train-Test Split")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt.gcf())

    features = ["Close_lag1", "Volume_lag1"]
    target = ["Close"]

    model = xgb.XGBRegressor()
    model.fit(train_data[features], train_data[target])
    predictions = model.predict(test_data[features])

    accuracy = model.score(test_data[features], test_data[target])

    plt.figure()
    plt.plot(test_data[target], label = "Actual")
    plt.plot(test_data.index, predictions, label = "Predicted")
    plt.title("Actual vs Predicted on Test Data")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt.gcf())

    st.write("Accuracy of the Model: ", accuracy)

    plt.figure()
    plt.plot(train_data["Close"], label = "Training Data")
    plt.plot(test_data[target], label = "Actual")
    plt.plot(test_data.index, predictions, label = "Predicted")
    plt.title("Actual vs Predicted")
    plt.xlabel("Date")
    plt.xticks(rotation=90)
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(plt.gcf())


    st.write("© Krish Jariwala. All Rights Reserved.")


