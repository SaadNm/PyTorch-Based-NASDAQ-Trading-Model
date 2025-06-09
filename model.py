import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import talib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ======================
# 1. MT5 Data Fetching
# ======================
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed!")
        mt5.shutdown()
        return False
    return True

def fetch_ustec_data(timeframe=mt5.TIMEFRAME_H4, n_bars=5000):
    if not initialize_mt5():
        return None
    
    rates = mt5.copy_rates_from_pos("USTEC", timeframe, 0, n_bars)
    mt5.shutdown()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# ======================
# 2. Feature Engineering
# ======================
def add_features(df):
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['log_ret'].rolling(14).std()
    
    # Moving Averages
    df['ma_10'] = talib.SMA(df['close'], timeperiod=10)
    df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
    
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # MACD
    macd, macdsignal, _ = talib.MACD(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    
    # Drop NaNs
    df.dropna(inplace=True)
    return df

# ======================
# 3. PyTorch Data Prep
# ======================
class TradingDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length, :-1]  # Features (all columns except last)
        y = self.data[idx+self.seq_length, -1]       # Target (last column: returns)
        return torch.FloatTensor(x), torch.FloatTensor([y])

# ======================
# 4. LSTM Model (PyTorch)
# ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last sequence output
        return out.squeeze()

# ======================
# 5. Sharpe Ratio Loss
# ======================
def sharpe_ratio_loss(preds, targets):
    returns = preds * targets  # Assume preds are directional signals (-1 to 1)
    sharpe = torch.mean(returns) / (torch.std(returns) + 1e-9)
    return -sharpe  # Minimize negative Sharpe = Maximize Sharpe

# ======================
# 6. Training Loop
# ======================
def train_model(model, dataloader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_sharpe = -np.inf
    
    for epoch in range(epochs):
        model.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = sharpe_ratio_loss(preds, y)
            loss.backward()
            optimizer.step()
        
        # Validation Sharpe
        val_sharpe = evaluate_sharpe(model, dataloader)
        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        
        print(f"Epoch {epoch+1}, Sharpe: {val_sharpe:.4f}")

def evaluate_sharpe(model, dataloader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            preds.append(model(x))
            targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    returns = preds * targets
    sharpe = torch.mean(returns) / (torch.std(returns) + 1e-9)
    return sharpe.item()

# ======================
# 7. Main Execution
# ======================
if __name__ == "__main__":
    # Fetch and preprocess data
    df = fetch_ustec_data()
    df = add_features(df)
    
    # Prepare target (next return)
    df['target_return'] = df['returns'].shift(-1)
    df.dropna(inplace=True)
    
    # Split features (X) and target (y)
    features = df.drop(['target_return', 'returns', 'log_ret'], axis=1)
    target = df['target_return']
    
    # Create dataset
    dataset = TradingDataset(pd.concat([features, target], axis=1).values)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LSTMModel(input_size=features.shape[1])
    train_model(model, dataloader)
    
    print("Training complete! Best model saved to 'best_lstm_model.pth'.")
