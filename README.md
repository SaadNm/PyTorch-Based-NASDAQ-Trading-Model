LSTM-Based Algorithmic Trading Model for USTEC
This project is an early attempt to use a Long Short-Term Memory (LSTM) neural network to predict market movements for USTEC (which is probably like the NASDAQ 100 index). The main goal was to build a basic model that could be trained to get the best Sharpe Ratio. This ratio helps you see how good your returns are compared to the risks you take.

As a student, this was one of my early attempts to use deep learning for trading. Even though there are more advanced models now, this project shows the core steps: getting data, making it useful, building a model, and training it with a specific financial aim.

What Does This Code Do?
Gets Data: It connects to MetaTrader 5 (MT5) to download historical 4-hour (H4) price data for USTEC.
Adds Features: It calculates important financial indicators from the raw price data. These include things like how much prices have changed, how volatile they are, and popular indicators like Simple Moving Averages (SMA), Relative Strength Index (RSI), and MACD.
Uses an LSTM Model: It builds a neural network called an LSTM. LSTMs are good at learning from data that comes in a sequence, like stock prices over time.
Learns for Sharpe Ratio: The model is trained to improve the Sharpe Ratio. This means it tries to make predictions that lead to good profits without taking on too much risk.
Saves the Best Model: During training, the code saves the version of the model that performs the best based on the Sharpe Ratio.

How to Run It
What You Need
Make sure you have:

Python 3.x
MetaTrader 5 software running on your computer (it needs to be open to get data).
These Python libraries. You can install them using pip:
Bash

pip install numpy pandas MetaTrader5 ta-lib torch scikit-learn
Steps to Run
Start the script: Open your command line or terminal, go to where you saved the Python file, and type:
Bash
your_script_name.py

The script will then fetch the data, prepare it, train the model, and save the best model as best_lstm_model.pth.

My Thoughts on This Project
This project was a great way for me to learn how to apply deep learning to financial markets.
