import datetime as dt
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, send_file
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
HISTORY_FILE = os.path.join(BASE_DIR, "recent_searches.json")
MODEL_PATH = os.path.join(BASE_DIR, "stock_dl_model.h5")
os.makedirs(STATIC_DIR, exist_ok=True)
app = Flask(__name__, static_folder=STATIC_DIR)


class LegacyCompatibleLSTM(LSTM):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("time_major", None)
        return super().from_config(config)


def load_legacy_model(model_path):
    try:
        return load_model(model_path, compile=False)
    except ValueError as exc:
        if "time_major" not in str(exc):
            raise
        return load_model(model_path, custom_objects={"LSTM": LegacyCompatibleLSTM}, compile=False)


model = None


def get_model():
    global model
    if model is None:
        model = load_legacy_model(MODEL_PATH)
    return model


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as history_file:
            data = json.load(history_file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_history(items):
    with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
        json.dump(items[:10], history_file, indent=2)


def add_recent_search(stock):
    history = [item for item in load_history() if item.get("ticker") != stock]
    history.insert(0, {"ticker": stock, "searched_at": dt.datetime.now().strftime("%d %b %Y, %I:%M %p")})
    save_history(history)


def get_stock_data(stock):
    df = yf.download(stock, start=dt.datetime(2000, 1, 1), end=dt.datetime(2024, 10, 1), auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def run_prediction(stock):
    df = get_stock_data(stock)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No historical data found for {stock}.")
    df = df.dropna(subset=["Close"]).copy()
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    ema100 = df["Close"].ewm(span=100, adjust=False).mean()
    ema200 = df["Close"].ewm(span=200, adjust=False).mean()

    split = int(len(df) * 0.70)
    train = pd.DataFrame(df["Close"].iloc[:split])
    test = pd.DataFrame(df["Close"].iloc[split:])
    if len(train) < 100 or test.empty:
        raise ValueError("Not enough historical data for the 100-day prediction window.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)
    final_df = pd.concat([train.tail(100), test], ignore_index=True)
    input_data = scaler.transform(final_df)
    x_test, y_test = [], []
    for index in range(100, len(input_data)):
        x_test.append(input_data[index - 100:index])
        y_test.append(input_data[index, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_pred = scaler.inverse_transform(get_model().predict(x_test, verbose=0))
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", stock)
    charts = {}
    chart_specs = [
        ("ema2050", f"{stock} - Closing Price with EMA 20 & EMA 50", [df["Close"], ema20, ema50], ["Closing Price", "EMA 20", "EMA 50"], "ema_20_50"),
        ("ema100200", f"{stock} - Closing Price with EMA 100 & EMA 200", [df["Close"], ema100, ema200], ["Closing Price", "EMA 100", "EMA 200"], "ema_100_200"),
    ]
    for key, title, series, labels, suffix in chart_specs:
        fig, axis = plt.subplots(figsize=(12, 6))
        for values, label in zip(series, labels):
            axis.plot(df.index, values, label=label)
        axis.set_title(title); axis.set_xlabel("Time"); axis.set_ylabel("Price"); axis.legend(); fig.tight_layout()
        charts[key] = f"{safe}_{suffix}.png"
        fig.savefig(os.path.join(STATIC_DIR, charts[key]), dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(y_actual, label="Actual Price", linewidth=1); axis.plot(y_pred, label="Predicted Price", linewidth=1)
    axis.set_title(f"{stock} - Prediction vs Actual"); axis.set_xlabel("Test Sequence"); axis.set_ylabel("Price"); axis.legend(); fig.tight_layout()
    charts["prediction"] = f"{safe}_prediction.png"
    fig.savefig(os.path.join(STATIC_DIR, charts["prediction"]), dpi=140); plt.close(fig)

    csv_name = f"{safe}_dataset.csv"
    df.to_csv(os.path.join(STATIC_DIR, csv_name))
    add_recent_search(stock)
    return {"stock": stock, "charts": charts, "dataset": csv_name, "rows": len(df), "start": df.index.min().strftime("%d %b %Y"), "end": df.index.max().strftime("%d %b %Y"), "data_desc": df.describe().to_html(classes="data-table", border=0)}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", page="home", recent_searches=load_history(), result=None, error=None)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    stock = request.form.get("stock", "").strip().upper() if request.method == "POST" else request.args.get("ticker", "").strip().upper()
    if not stock:
        stock = "POWERGRID.NS" if request.method == "POST" else ""
    if not stock:
        return render_template("index.html", page="analyze", recent_searches=load_history(), result=None, error=None)
    if not re.fullmatch(r"[A-Z0-9^._-]{1,20}", stock):
        return render_template("index.html", page="analyze", recent_searches=load_history(), result=None, error="Enter a valid stock ticker, e.g. POWERGRID.NS."), 400
    try:
        result = run_prediction(stock)
        return render_template("index.html", page="analyze", recent_searches=load_history(), result=result, error=None)
    except Exception as exc:
        return render_template("index.html", page="analyze", recent_searches=load_history(), result=None, error=str(exc)), 404


@app.route("/history")
def history():
    return render_template("index.html", page="history", recent_searches=load_history(), result=None, error=None)


@app.route("/about")
def about():
    return render_template("index.html", page="about", recent_searches=load_history(), result=None, error=None)


@app.route("/api/recent-searches/clear", methods=["POST"])
def clear_recent_searches():
    save_history([])
    return jsonify({"success": True})

@app.route('/download/<filename>')
def download_file(filename):
    if os.path.basename(filename) != filename:
        return "Invalid file name", 400
    path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8080")), debug=True, use_reloader=False)
