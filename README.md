# 📈 StockSense — LSTM-Powered Stock Price Prediction

> Predicting tomorrow's price, one candle at a time — powered by Deep Learning and served through Flask.

StockSense is an end-to-end stock market forecasting web app. It pulls years of historical stock data, engineers technical indicators used by real traders (moving averages, EMAs), and feeds them into a stacked LSTM neural network to predict future closing prices. The trained model is wrapped in a lightweight Flask app so predictions are just a browser tab away.

---

## 🚀 What It Does

- **Fetches live historical data** for any stock ticker (default: `POWERGRID.NS`) straight from Yahoo Finance, spanning 2000–2024 (4,200+ trading days).
- **Visualizes market behavior** through closing price, opening price, high price, volume trends, and candlestick charts.
- **Engineers technical indicators**: 100-day & 200-day Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) — the same signals used in real technical analysis.
- **Trains a Deep Learning model** — a 4-layer stacked LSTM network — to learn temporal patterns in price movement.
- **Serves predictions through a Flask web app**, so results are viewable in a browser instead of a notebook.

---

## 🧠 How It Works

### 1. Data Collection
Historical OHLCV (Open, High, Low, Close, Volume) data is downloaded using `yfinance` for the ticker `POWERGRID.NS`, covering **Jan 2000 – Nov 2024**.

### 2. Exploratory Data Analysis
- Null-value checks, statistical summaries, and shape inspection.
- Interactive **candlestick charts** (via Plotly) to visualize price action.
- Line plots for Close, Open, High, and Volume trends over time.

### 3. Feature Engineering
- **Simple Moving Averages**: 100-day and 200-day rolling means, plotted against the closing price to spot trend crossovers.
- **Exponential Moving Averages**: 100-day and 200-day EMAs, which react faster to recent price changes than SMAs.

### 4. Preprocessing
- Data split **70% training / 30% testing** (chronological split — no shuffling, since order matters in time series).
- Features scaled to a **[0, 1]** range using `MinMaxScaler`.
- Sequences built with a **100-day lookback window**: each training sample uses the previous 100 days of prices to predict the next day's close.

### 5. Model Architecture
A deep, stacked LSTM network built with Keras:

| Layer | Units | Activation | Dropout |
|-------|-------|------------|---------|
| LSTM 1 | 50  | ReLU | 0.2 |
| LSTM 2 | 60  | ReLU | 0.3 |
| LSTM 3 | 80  | ReLU | 0.4 |
| LSTM 4 | 120 | ReLU | 0.5 |
| Dense (Output) | 1 | — | — |

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 50
- Increasing dropout at each layer (0.2 → 0.5) progressively fights overfitting as the network gets deeper.

### 6. Evaluation
The trained model predicts prices on the unseen test set, which are then de-scaled back to real price values and plotted against the actual prices to visually assess accuracy.

### 7. Deployment
The trained model is exported (`stock_dl_model.h5`) and loaded into a **Flask** web application, which takes a ticker symbol as input and returns visualized predictions in the browser.

---

## 🗂️ Project Structure

```
StockSense/
├── app.py                     # Flask application — routes & prediction logic
├── stock_dl_model.h5          # Trained LSTM model
├── Stock_Price_Prediction_.ipynb  # Model research & training notebook
├── templates/
│   └── index.html             # Web UI for entering a ticker & viewing results
├── static/
│   └── (CSS / charts / assets)
├── requirements.txt
└── README.md
```
> Note: this reflects the standard layout for this kind of Flask + Keras project — adjust filenames above to match your actual `app.py`/`templates` if they differ.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | Keras / TensorFlow (LSTM) |
| Data | yfinance, pandas, NumPy |
| Preprocessing | scikit-learn (`MinMaxScaler`) |
| Visualization | Matplotlib, Plotly |
| Web Framework | Flask |

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/StockSense.git
cd StockSense

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

**Suggested `requirements.txt`:**
```
flask
numpy
pandas
matplotlib
plotly
yfinance
scikit-learn
keras
tensorflow
```

---

## 💻 Usage

1. Launch the app and open it in your browser.
2. Enter a stock ticker (e.g. `POWERGRID.NS`, `AAPL`, `TCS.NS`).
3. View the historical price chart, moving averages, and the model's predicted vs. actual price plot.

---

## 📊 Results

The model successfully captures the **overall trend and momentum** of the stock's price movement, closely tracking the actual closing prices on the test set — demonstrating that even a moderately deep LSTM can learn meaningful temporal patterns from 100-day price windows.

*(Add your prediction-vs-actual chart screenshot here for the GitHub preview — it makes a strong visual for recruiters skimming your repo.)*

---

## 🔮 Future Improvements

- Add more features beyond Close price (Volume, RSI, MACD) as model inputs.
- Support multi-ticker comparison in a single view.
- Add confidence intervals around predictions.
- Deploy publicly (Render / Railway / HuggingFace Spaces) with a live demo link.
- Replace the manual scaler-factor calculation with `scaler.inverse_transform()` for robustness.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price prediction using historical data does not account for real-world market volatility, news events, or macroeconomic factors. **Do not use this model for actual financial/trading decisions.**

---

## 👤 Author

**Arsalan shahid**
BCA (AI & Data Analytics) — Teerthanker Mahaveer University