import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

# --- Data Functions ---

def get_stock_data(symbol):
    end = datetime.today()
    start = end - timedelta(days=365 * 10)
    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty or 'Close' not in data.columns:
        raise ValueError(f"No data found for {symbol}")
    data = data[['Close']].reset_index()
    data.columns = ['ds', 'y']
    return data

def forecast_stock(symbol):
    data = get_stock_data(symbol)
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=365 * 10)
    forecast = model.predict(future)
    return data, forecast, model

def plot_forecast_with_continuous_line(symbol):
    data, forecast, model = forecast_stock(symbol)
    today = pd.Timestamp(datetime.today().date())

    try:
        company_info = yf.Ticker(symbol).info
        company_name = company_info.get("shortName", symbol.upper())
    except Exception:
        company_name = symbol.upper()

    actual_today_price = data[data['ds'] == today]['y']
    if actual_today_price.empty:
        actual_today_price = data['y'].iloc[-1]
    else:
        actual_today_price = actual_today_price.values[0]

    predicted_today_price = forecast[forecast['ds'] == today]['yhat']
    if predicted_today_price.empty:
        predicted_today_price = np.interp(today.timestamp(),
                                          forecast['ds'].map(pd.Timestamp.timestamp),
                                          forecast['yhat'])
    else:
        predicted_today_price = predicted_today_price.values[0]

    adjustment = actual_today_price - predicted_today_price
    forecast['yhat_adj'] = forecast['yhat'] + adjustment

    plt.figure(figsize=(10, 6))
    past_data = data[data['ds'] <= today]
    future_forecast = forecast[forecast['ds'] >= today]

    plt.plot(past_data['ds'], past_data['y'], label='Actual Price', color='blue')
    plt.plot(future_forecast['ds'], future_forecast['yhat_adj'], label='Predicted Price', color='orange')
    plt.axvline(x=today, color='yellow', linestyle='--', linewidth=2, label='Today')

    plt.title(f'{company_name} ({symbol.upper()}) Stock Price Forecast (Continuous)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD $)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_stocks(tickers):
    if len(tickers) < 2:
        messagebox.showinfo("Info", "Please select at least two stocks to compare.")
        return

    plt.figure(figsize=(12, 7))
    today = pd.Timestamp(datetime.today().date())

    for symbol in tickers:
        try:
            data, forecast, model = forecast_stock(symbol)
            company_info = yf.Ticker(symbol).info
            name = company_info.get("shortName", symbol.upper())
            predicted = forecast[['ds', 'yhat']]
            plt.plot(predicted['ds'], predicted['yhat'], label=f'{name} ({symbol.upper()})')
        except Exception as e:
            print(f"Error comparing {symbol}: {e}")

    plt.axvline(x=today, color='yellow', linestyle='--', linewidth=2, label='Today')
    plt.title("Stock Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price (USD $)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_pct_change(ticker, days):
    try:
        end = datetime.today()
        start = end - timedelta(days=days + 1)
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
        if df.empty or 'Close' not in df.columns:
            return None
        df = df.sort_index()
        if len(df) < 2:
            return None
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        pct_change = ((end_price - start_price) / start_price) * 100
        return pct_change
    except Exception:
        return None

def fmt(val):
    if val is None:
        return "N/A", "black"
    if isinstance(val, (pd.Series, np.ndarray)):
        if len(val) == 0:
            return "N/A", "black"
        val = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val[0])
    color = "green" if val > 0 else ("red" if val < 0 else "black")
    return f"{val:+.2f}%", color

# --- Favorites Persistence ---

FAV_FILE = "favorites.json"

def save_favorites():
    with open(FAV_FILE, "w") as f:
        json.dump(favorites, f)

def load_favorites():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# --- GUI Functions ---

def on_predict():
    ticker = ticker_entry.get().strip().upper()
    if not ticker:
        messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
        return
    try:
        plot_forecast_with_continuous_line(ticker)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def add_favorite():
    ticker = fav_entry.get().strip().upper()
    if not ticker:
        messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
        return
    if ticker in favorites:
        messagebox.showinfo("Info", f"{ticker} is already in favorites.")
        return
    if len(favorites) >= 10:
        messagebox.showwarning("Limit reached", "You can only add up to 10 favorites.")
        return
    try:
        info = yf.Ticker(ticker).info
        if 'shortName' not in info:
            raise ValueError
    except Exception:
        messagebox.showerror("Error", f"Could not find company info for {ticker}.")
        return
    favorites.append(ticker)
    save_favorites()
    refresh_favorites_tree()
    fav_entry.delete(0, tk.END)

def remove_favorite():
    selected = fav_tree.selection()
    if not selected:
        messagebox.showinfo("Info", "Please select a favorite to remove.")
        return
    for sel in selected:
        ticker = fav_tree.item(sel, 'values')[1]
        if ticker in favorites:
            favorites.remove(ticker)
    save_favorites()
    refresh_favorites_tree()

def refresh_favorites_tree():
    for i in fav_tree.get_children():
        fav_tree.delete(i)
    for ticker in favorites:
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName", ticker)
        except Exception:
            name = ticker
        day_change = get_pct_change(ticker, 1)
        week_change = get_pct_change(ticker, 7)
        month_change = get_pct_change(ticker, 30)
        year_change = get_pct_change(ticker, 365)
        day_str, day_col = fmt(day_change)
        week_str, week_col = fmt(week_change)
        month_str, month_col = fmt(month_change)
        year_str, year_col = fmt(year_change)
        fav_tree.insert('', 'end', values=(name, ticker, day_str, week_str, month_str, year_str),
                        tags=(day_col,))
    fav_tree.tag_configure('green', foreground='green')
    fav_tree.tag_configure('red', foreground='red')
    fav_tree.tag_configure('black', foreground='white')

def on_tab_changed(event):
    tab = event.widget.tab('current')['text']
    if tab == "Favorites":
        refresh_favorites_tree()

def on_compare():
    selected = fav_tree.selection()
    tickers = [fav_tree.item(sel, 'values')[1] for sel in selected]
    compare_stocks(tickers)

# --- Main GUI Setup ---

root = tk.Tk()
root.title("Stock Predictor")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
root.configure(bg="#1f1f2e")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Predictor Tab
predictor_frame = tk.Frame(notebook, bg="#1f1f2e")
notebook.add(predictor_frame, text="Predictor")

title_label = tk.Label(predictor_frame, text="📈 Stock Predictor", font=("Helvetica", 48, "bold"), fg="#ffcb6b", bg="#1f1f2e")
title_label.pack(pady=(40, 40))

instruction_label = tk.Label(predictor_frame, text="Enter Stock Ticker Symbol:", font=("Helvetica", 24), fg="#c792ea", bg="#1f1f2e")
instruction_label.pack(pady=(0, 20))

ticker_entry = tk.Entry(predictor_frame, font=("Helvetica", 24), width=20, justify='center')
ticker_entry.pack(pady=10)

predict_button = tk.Button(predictor_frame, text="Predict", font=("Helvetica", 24, "bold"), bg="#82aaff", fg="#1f1f2e",
                           activebackground="#4f78ff", activeforeground="#ffffff", command=on_predict)
predict_button.pack(pady=30)

# Favorites Tab
favorites_frame = tk.Frame(notebook, bg="#1f1f2e")
notebook.add(favorites_frame, text="Favorites")

fav_controls = tk.Frame(favorites_frame, bg="#1f1f2e")
fav_controls.pack(pady=10)

fav_entry = tk.Entry(fav_controls, font=("Helvetica", 16), width=10, justify='center')
fav_entry.pack(side=tk.LEFT, padx=10)

fav_add_btn = tk.Button(fav_controls, text="Add Favorite", font=("Helvetica", 14), bg="#82aaff", fg="#1f1f2e",
                        activebackground="#4f78ff", activeforeground="#ffffff", command=add_favorite)
fav_add_btn.pack(side=tk.LEFT, padx=10)

fav_remove_btn = tk.Button(fav_controls, text="Remove Selected", font=("Helvetica", 14), bg="#d74a49", fg="white",
                           activebackground="#a93634", activeforeground="white", command=remove_favorite)
fav_remove_btn.pack(side=tk.LEFT, padx=10)

compare_btn = tk.Button(fav_controls, text="Compare Selected", font=("Helvetica", 14), bg="#c792ea", fg="#1f1f2e",
                        activebackground="#a47de0", activeforeground="#ffffff", command=on_compare)
compare_btn.pack(side=tk.LEFT, padx=10)

compare_note = tk.Label(favorites_frame, text="* Hold Ctrl (Cmd on Mac) to select multiple stocks for comparison.",
                        font=("Helvetica", 12), fg="lightgrey", bg="#1f1f2e")
compare_note.pack()

columns = ("Company", "Ticker", "24h Change", "7d Change", "30d Change", "1yr Change")
fav_tree = ttk.Treeview(favorites_frame, columns=columns, show='headings', height=15)
for col in columns:
    fav_tree.heading(col, text=col)
    fav_tree.column(col, width=150, anchor=tk.CENTER)
fav_tree.pack(padx=20, pady=20, fill='both', expand=True)

style = ttk.Style()
style.theme_use("default")
style.configure("Treeview", background="#2e2e3e", foreground="white", rowheight=30,
                fieldbackground="#2e2e3e", font=("Helvetica", 12))
style.map('Treeview', background=[('selected', '#4f78ff')])
style.configure("Treeview.Heading", font=("Helvetica", 14, "bold"), background="#1f1f2e", foreground="#ffcb6b")

favorites = load_favorites()
notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
root.mainloop()
