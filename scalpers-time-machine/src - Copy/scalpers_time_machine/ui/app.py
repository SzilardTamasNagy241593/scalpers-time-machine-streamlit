import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import logging
from scalpers_time_machine.model import train_model
from scalpers_time_machine.metrics import evaluate_model
from scalpers_time_machine.data_loader import load_data
from scalpers_time_machine.config import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)

# Preload data and model
df_all = load_data()
df_all.index = pd.to_datetime(df_all.index).tz_localize(None)
min_year = df_all.index.min().year
max_year = df_all.index.max().year
year_choices = [str(y) for y in range(min_year, max_year + 1)]
month_choices = [f"{m:02d}" for m in range(1, 13)]
day_choices = [f"{d:02d}" for d in range(1, 32)]
tickers = sorted(df_all["company_prefix"].unique().tolist())
ticker_choices = ["All"] + tickers

model, X_test, y_test = train_model()

def predict_and_plot(stock_filter, start_year, start_month, start_day,
                     end_year, end_month, end_day, investment_amount):

    df = load_data()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    try:
        investment_amount = float(investment_amount)
        if investment_amount <= 0:
            raise ValueError("Investment must be greater than 0.")
    except Exception as e:
        return f"<span style='color:red;'>Invalid investment amount: {e}</span>", go.Figure(), {"error": str(e)}, go.Figure()

    try:
        start_dt = pd.to_datetime(f"{start_year}-{start_month}-{start_day}")
        end_dt = pd.to_datetime(f"{end_year}-{end_month}-{end_day}")
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    except Exception as e:
        return f"<span style='color:red;'>Invalid date selection: {e}</span>", go.Figure(), {"error": str(e)}, go.Figure()

    if stock_filter != "All":
        df = df[df["company_prefix"].str.upper() == stock_filter.upper()]

    if df.empty:
        return f"<span style='color:red;'>No data available for {stock_filter} in selected range.</span>", go.Figure(), {}, go.Figure()

    df = df.dropna(subset=FEATURE_COLUMNS)
    df = df.sort_index()

    try:
        preds = model.predict(df[FEATURE_COLUMNS])
        df["prediction"] = preds
    except Exception as e:
        logging.error("Prediction failed:", exc_info=True)
        return f"<span style='color:red;'>Prediction failed: {e}</span>", go.Figure(), {}, go.Figure()

    # Price chart
    fig_price = go.Figure()
    for ticker in df["company_prefix"].unique():
        ticker_df = df[df["company_prefix"] == ticker].sort_index()
        fig_price.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df["close_value"], mode="lines", name=ticker))
    fig_price.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Close Price")

    metrics = evaluate_model(model, X_test, y_test)

    # Strategy simulation
    fig_strategy = go.Figure()
    if stock_filter != "All":
        test_df = df[df["company_prefix"] == stock_filter].copy().sort_index()
        if not test_df.empty:
            cash = investment_amount
            position = 0
            portfolio_values = []
            buy_hold_values = []
            initial_price = test_df["close_value"].iloc[0]

            for i in range(len(test_df)):
                pred = test_df["prediction"].iloc[i]
                price = test_df["close_value"].iloc[i]

                if pred > 0.001 and position == 0:
                    position = cash / price
                    cash = 0
                elif pred < -0.001 and position > 0:
                    cash = position * price
                    position = 0

                value = cash if position == 0 else position * price
                portfolio_values.append(value)
                bh_value = investment_amount * (price / initial_price)
                buy_hold_values.append(bh_value)

            fig_strategy.add_trace(go.Scatter(
                x=test_df.index,
                y=buy_hold_values,
                name="Buy & Hold",
                line=dict(color="green"),
                legendrank=2
            ))
            fig_strategy.add_trace(go.Scatter(
                x=test_df.index,
                y=portfolio_values,
                name="Strategy",
                line=dict(dash="dash", color="gray"),
                legendrank=1
            ))
            fig_strategy.update_layout(title="Strategy vs Buy-and-Hold", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    else:
        fig_strategy.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=["Strategy simulation only for single stocks."], textposition="middle center"))
        fig_strategy.update_layout(xaxis_visible=False, yaxis_visible=False)

    try:
        first = df["open_value"].iloc[0]
        last = df["close_value"].iloc[-1]
        shares = investment_amount / first
        final_value = shares * last
        gain = final_value - investment_amount
        pct_return = (gain / investment_amount) * 100

        final_str = f"${final_value:,.2f}"
        gain_str = f"+${gain:,.2f}" if gain > 0 else f"-${abs(gain):,.2f}"
        pct_str = f"(+{pct_return:.2f}%)" if gain > 0 else f"(-{abs(pct_return):.2f}%)"
        gain_color = "green" if gain > 0 else "red"

        result_html = f"<span style='color:{gain_color}; font-size: 18px;'>Return: {gain_str} {pct_str} → Final Value: {final_str}</span>"
    except Exception as e:
        result_html = f"<span style='color:red;'>Error calculating return: {e}</span>"

    return result_html, fig_price, metrics, fig_strategy

def help_section():
    return """
    ## Help Guide

    Welcome to Scalper's Time Machine — a tool for simulating AI-powered stock trend predictions and comparing them to a Buy & Hold strategy.

    ### Inputs (left panel)

    - **Stock Ticker:**  
      Choose a specific stock (e.g., AAPL, GOOG) or select "All" to view every available ticker in the database.

    - **Start Date / End Date:**  
      Define the historical time window for your analysis by selecting the year, month, and day. The model will run simulations using data within this date range.

    - **Investment Amount (USD):**  
      Enter the amount you'd hypothetically invest. This value will scale both the AI strategy and Buy & Hold portfolio simulations accordingly.

    - **Submit Button:**  
      Executes the model, generates predictions, and updates all charts and metrics based on the selected inputs.

    - **Clear Button:**  
      Resets all selections and clears the result output.

    ### Outputs (right panel)

    - **Stock Price Chart:**  
      Displays the closing prices of the selected stock(s) over the selected date range. Useful for identifying trends and price movements.

    - **Model Evaluation:**  
      Shows test performance metrics for the machine learning model:
        - `mae`: Mean Absolute Error — average magnitude of prediction errors
        - `rmse`: Root Mean Squared Error — punishes larger errors more
        - `r2`: R-squared — proportion of variance explained
        - `accuracy`, `precision`, `recall`, `f1_score`: Classification performance when converting return predictions into binary up/down trends

    - **Strategy vs Buy-and-Hold Chart:**  
      Compares two strategies:
        - **Strategy (green line):** Simulates trading based on the model's predictions (buy when return is predicted to go up, sell when predicted to go down)
        - **Buy & Hold (gray dashed line):** Buys at the start and holds until the end of the date range

    - **Return Summary:**  
      Shows the total return from the Buy & Hold strategy based on your investment amount. Includes net gain/loss and percentage return.

    ### Important Notes

    This tool is designed for educational and experimental purposes only. It demonstrates how machine learning can be applied to financial data but does not guarantee performance or financial return. Always do your own research before making investment decisions.
    """

# Gradio UI definition
with gr.Blocks() as home_ui:
    gr.Markdown("## Scalper's Time Machine")
    gr.Markdown("AI-based stock trend predictor with investment simulation")
    gr.Markdown(f"Data available from {min_year} to {max_year}.")

    with gr.Row():
        with gr.Column():
            stock = gr.Dropdown(choices=ticker_choices, label="Stock Ticker", value="All")
            gr.Markdown("Start Date")
            with gr.Row():
                start_year = gr.Dropdown(choices=year_choices, label="Year", value=str(min_year))
                start_month = gr.Dropdown(choices=month_choices, label="Month", value="01")
                start_day = gr.Dropdown(choices=day_choices, label="Day", value="01")
            gr.Markdown("End Date")
            with gr.Row():
                end_year = gr.Dropdown(choices=year_choices, label="Year", value=str(max_year))
                end_month = gr.Dropdown(choices=month_choices, label="Month", value="12")
                end_day = gr.Dropdown(choices=day_choices, label="Day", value="31")
            amount = gr.Textbox(label="Investment Amount (USD)", value="10000")
            submit = gr.Button("Submit")
            clear = gr.Button("Clear")
            result_html = gr.HTML(label="Return Summary")

        with gr.Column():
            chart = gr.Plot(label="Stock Price Chart")
            metrics = gr.JSON(label="Model Evaluation")
            strategy_chart = gr.Plot(label="Strategy vs Buy-and-Hold")

    submit.click(fn=predict_and_plot,
                 inputs=[stock, start_year, start_month, start_day,
                         end_year, end_month, end_day, amount],
                 outputs=[result_html, chart, metrics, strategy_chart])

    clear.click(fn=lambda: ("", go.Figure(), {}, go.Figure()),
                inputs=[], outputs=[result_html, chart, metrics, strategy_chart])

with gr.Blocks() as help_ui:
    gr.Markdown(help_section())

tabs = gr.TabbedInterface([home_ui, help_ui], tab_names=["Home", "Help"])

if __name__ == "__main__":
    print("Launching dashboard at http://127.0.0.1:7860")
    tabs.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
