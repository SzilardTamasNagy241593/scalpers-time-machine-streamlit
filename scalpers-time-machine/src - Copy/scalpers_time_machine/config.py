TABLE_NAME = "stock_daily_enriched"

FEATURE_COLUMNS = [
    "open_value",
    "high_value",
    "low_value",
    "close_value",
    "return_pct_lag_1",
    "intraday_volatility",
    "intraday_volatility_lag_1",
    "intraday_volatility_lag_2",
    "intraday_volatility_lag_3",
    "return_pct_rolling_mean_5",
    "return_pct_rolling_std_5",
    "return_pct_rolling_mean_10",
    "return_pct_rolling_std_10",
    "intraday_volatility_rolling_mean_5",
    "intraday_volatility_rolling_std_5",
    "intraday_volatility_rolling_mean_10",
    "intraday_volatility_rolling_std_10",
    "best_buy_time_min",
    "best_sell_time_min"
]

TARGET_COLUMN = "return_pct"