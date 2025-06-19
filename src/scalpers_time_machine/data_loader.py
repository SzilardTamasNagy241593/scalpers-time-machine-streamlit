import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
logging.basicConfig(level=logging.INFO)

TABLE_NAME = "stock_daily_enriched"
TARGET_COLUMN = "return_pct"

def time_to_minutes(t):
    """Convert HH:MM time strings to minutes since midnight."""
    if pd.isnull(t):
        return -1
    if isinstance(t, str):
        parts = t.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    return t.hour * 60 + t.minute

def load_data():
    """Load, preprocess, and return cleaned stock data from the database or CSV."""
    db_url = os.getenv("DATABASE_URL")
    try:
        if not db_url:
            raise ValueError("DATABASE_URL is not set")
        engine = create_engine(db_url)
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql(query, con=engine, index_col="date_value", parse_dates=["date_value"])
        logging.info("Data loaded from PostgreSQL")
    except Exception as e:
        logging.warning(f"DB load failed: {e}. Falling back to CSV.")
        df = pd.read_csv("data/stock_daily_enriched.csv", index_col="date_value", parse_dates=["date_value"])

    # Convert times to minutes
    for col in ['best_buy_time', 'best_sell_time']:
        if col in df.columns:
            df[f"{col}_min"] = df[col].apply(time_to_minutes)
            df.drop(columns=[col], inplace=True)

    # Create volatility feature
    df["intraday_volatility"] = df["high_value"] - df["low_value"]

    # Lag features
    df["return_pct_lag_1"] = df["return_pct"].shift(1)
    df["intraday_volatility_lag_1"] = df["intraday_volatility"].shift(1)
    df["intraday_volatility_lag_2"] = df["intraday_volatility"].shift(2)
    df["intraday_volatility_lag_3"] = df["intraday_volatility"].shift(3)

    # Rolling window features (5 and 10 days)
    df["return_pct_rolling_mean_5"] = df["return_pct"].rolling(5).mean()
    df["return_pct_rolling_std_5"] = df["return_pct"].rolling(5).std()
    df["return_pct_rolling_mean_10"] = df["return_pct"].rolling(10).mean()
    df["return_pct_rolling_std_10"] = df["return_pct"].rolling(10).std()

    df["intraday_volatility_rolling_mean_5"] = df["intraday_volatility"].rolling(5).mean()
    df["intraday_volatility_rolling_std_5"] = df["intraday_volatility"].rolling(5).std()
    df["intraday_volatility_rolling_mean_10"] = df["intraday_volatility"].rolling(10).mean()
    df["intraday_volatility_rolling_std_10"] = df["intraday_volatility"].rolling(10).std()

    # Drop missing values caused by shifting and rolling
    df.dropna(inplace=True)

    return df