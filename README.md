# LSTM Stock Price Prediction

## Overview
This project applies a Long Short-Term Memory (LSTM) neural network to predict stock prices, leveraging the LSTM's ability to learn order dependencies in sequence prediction problems. We predict the next-day stock prices by training on historical data, testing univariate and multivariate LSTM models, and using various optimizers.

## Prerequisites
- Python 3.x
- Jupyter Notebook or Google Colab
- TensorFlow
- NumPy
- Pandas
- Matplotlib

## Installation
Clone this repository to your local machine and install dependencies:

```bash
git clone https://github.com/Serendeep/Multivariate-Univariate-LSTM.git
cd Multivariate-Univariate-LSTM
pip install -r requirements.txt
```

## Dataset
The dataset consists of 23 years of stock data from Carriage Services, Inc. (CSV), including open, high, low, close prices, and volume. Data preprocessing steps include cleaning, handling missing values, and normalization.

## Model Architecture
The LSTM model architecture includes:
- Univariate LSTM: uses only closing prices as input.
- Multivariate LSTM: uses closing price, high price, and volume.

The model also integrates regularization techniques and optimizers to reduce overfitting and optimize prediction performance.

## Training
The model was trained with a 70:30 train-test split using the Adam and RMSprop optimizers to find the best performance. We experimented with different data split ratios (3:1, 4:1, etc.) to evaluate the impact on Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Results
- **Best Accuracy**: The 5:1 split yielded the highest accuracy in both Univariate (RMSE: 0.016373) and Multivariate LSTM (RMSE: 0.013424).
- **Visualization**: The model's performance was visualized, showing predicted vs. actual stock prices, using both univariate and multivariate methods.

## Future Work
Future improvements include experimenting with the number of epochs and incorporating sentiment analysis using social media data. We also plan to enhance feature engineering for better market trend analysis.

## References
1. Nusrat Rouf et al., "Stock Market Prediction Using Machine Learning Techniques"
2. Y. Dai and Y. Zhang, "Machine Learning in Stock Price Trend Forecasting," Stanford University
3. P. Nayyeri et al., "Deep Learning for Stock Market Prediction"
