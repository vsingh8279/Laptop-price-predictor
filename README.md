# Laptop Price Predictor

Predict the approximate price of a laptop based on specifications using machine learning techniques.

## Introduction
This project aims to predict the price of laptops based on their specifications using advanced machine learning algorithms. Various regression algorithms such as Random Forest, Gradient Boosting, XGBoost, and Extra Trees were explored, and a Voting Regressor was selected for the final model based on the highest R2 score and low MAE.

## How It Works
The project is built using Python and the Streamlit library for the user interface. The app allows users to input laptop specifications, including brand, type, RAM, weight, touchscreen, IPS, screen size, resolution, CPU, HDD, SSD, GPU, and OS. The trained model then predicts the price of the laptop configuration.

## Key Features
- Implemented a user-friendly web interface using Streamlit.
- Utilized various machine learning algorithms to build predictive models.
- Deployed the app on Streamlit Cloud for easy access and usage.

## How to Run
1. Install the required libraries using `pip install -r requirements.txt`.
2. Run the app using `streamlit run app.py`.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

## Author
Hemant Singh Meena

