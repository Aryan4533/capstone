import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Place your data preprocessing, model implementation, evaluation, and visualization code here.

def main():
    st.title("Machine Learning Model Selector")

    # Input: Select dataset
    st.subheader("Select a Dataset:")
    dataset_selector = st.selectbox("Choose a dataset:", ["Dataset 1", "Dataset 2", "Dataset 3"])  # Add your dataset names
    if dataset_selector == "Dataset 1":
        # Load and preprocess Dataset 1
        data = pd.read_csv("/content/NIFTY 50.csv")
        # Add your data preprocessing code
        data.dropna()
        # Set the Date column as the index and convert it to datetime
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data.set_index('Date', inplace=True)
        # Define features and target
        X = data[['Open', 'High', 'Low','Volume']].values
        y = data['Close'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Feature scaling if necessary
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
    # Input: Select machine learning model
    st.subheader("Select a Machine Learning Model:")
    model_selector = st.selectbox("Choose a model:", ["Model 1", "Model 2", "Model 3"])  # Add your model names
    if model_selector == "Model 1":
      # Implement and train Model 1
        # Add your model implementation and training code
      # Model 1: Feedforward Neural Network (Multilayer Perceptron)
      model1 = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
])

      model1.compile(optimizer='adam', loss='mean_squared_error')
      model1.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
      

    # Your existing code for evaluation and visualization
    def evaluate_model(model, X_test, y_test):
      y_pred = model.predict(X_test)
      mse = mean_squared_error(y_test, y_pred)
      r2 = r2_score(y_test, y_pred)
      return mse, r2

# Evaluate the models
    mse1, r2_1 = evaluate_model(model1, X_test, y_test)
    # ...

if __name__ == "__main__":
    main()
