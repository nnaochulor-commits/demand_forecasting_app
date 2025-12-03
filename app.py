import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from openai import OpenAI

# --------------- CONFIG & STYLING ---------------

st.set_page_config(page_title="SmartInventory AI", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #2A4D69;
    }
    .sub-title {
        font-size: 18px;
        color: #4F4F4F;
    }
    .block-container {padding-top: 1rem;}
    .stButton>button {
        background-color: #2A4D69;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stDownloadButton>button {
        background-color: #4F8A10;
        color: white;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<p class="main-title">SmartInventory AI — Demand Forecasting Assistant</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Upload your historical sales data and instantly get AI-powered demand forecasts and reorder suggestions.</p>',
    unsafe_allow_html=True,
)
st.write("---")

# Sidebar guide
st.sidebar.title("📘 Quick Guide")
st.sidebar.info(
    """
    1️⃣ Upload your sales data  
    2️⃣ Review 14-day demand & reorder table  
    3️⃣ Inspect detailed forecast per product  
    4️⃣ Generate AI-written business report  
    """
)
st.sidebar.write("💬 Contact:")
st.sidebar.write("smartinventory.ai@gmail.com")

# --------------- CONSTANTS ---------------

MIN_HISTORY_DAYS = 60        # minimum history per product to forecast
FORECAST_DAYS_SINGLE = 30    # single-product detailed forecast horizon
FORECAST_DAYS_MULTI = 14     # multi-product planning horizon (reorder)

# --------------- OPENAI CLIENT ---------------

# For Streamlit Cloud, put OPENAI_API_KEY in st.secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# --------------- FUNCTIONS: DATA & FORECASTING ---------------

def load_and_clean_data(file) -> pd.DataFrame:
    """
    Load CSV or Excel, enforce required columns, normalize, aggregate
    Returns df with columns: ['product_name', 'ds', 'y']
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_excel(file)

    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "product_name", "quantity"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)

    # aggregate to daily per product
    df = (
        df.groupby(["product_name", "date"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"date": "ds", "quantity": "y"})
    )

    return df


def generate_forecast(df: pd.DataFrame, product_name: str) -> pd.DataFrame | None:
    """
    Single-product detailed forecast using Prophet.
    Input df: ['product_name', 'ds', 'y'] for all products.
    Returns forecast df for this product or None if insufficient history.
    """
    product_df = df[df["product_name"] == product_name][["ds", "y"]].copy()
    if product_df.empty:
        return None

    product_df = product_df.sort_values("ds")
    history_days = (product_df["ds"].max() - product_df["ds"].min()).days

    if history_days < MIN_HISTORY_DAYS:
        return None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
    )
    model.fit(product_df)

    future = model.make_future_dataframe(periods=FORECAST_DAYS_SINGLE)
    forecast = model.predict(future)
    return forecast


def build_multi_product_forecast(df: pd.DataFrame, horizon_days: int = FORECAST_DAYS_MULTI) -> pd.DataFrame:
    """
    For each product, forecast next `horizon_days` total quantity.
    Returns df: [Product, Predicted Demand Next N Days]
    """
    results = []

    for product in df["product_name"].unique():
        product_df = df[df["product_name"] == product][["ds", "y"]].copy().sort_values("ds")
        if product_df.empty:
            continue

        history_days = (product_df["ds"].max() - product_df["ds"].min()).days
        if history_days < MIN_HISTORY_DAYS or product_df["y"].sum() == 0:
            continue

        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.5,
            )
            model.fit(product_df)
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future)

            total_forecast = forecast.tail(horizon_days)["yhat"].sum()
            results.append(
                [product, round(total_forecast, 1)]
            )
        except Exception:
            # skip problematic product
            continue

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(
        results, columns=["Product", f"Predicted Demand Next {horizon_days} Days"]
    )
    return df_results


# --------------- FUNCTIONS: GPT BUSINESS REPORT ---------------

def build_forecast_summary_table(df_results: pd.DataFrame, horizon_days: int) -> str:
    """
    Build a concise markdown table to send to GPT.
    """
    cols = ["Product", f"Predicted Demand Next {horizon_days} Days", "Current Stock", "Recommended Order", "Risk Status"]
    df_small = df_results[cols].copy()
    return df_small.to_markdown(index=False)


def generate_ai_business_report(df_results: pd.DataFrame, shop_name: str, horizon_days: int = FORECAST_DAYS_MULTI) -> str:
    """
    Use OpenAI GPT model to generate a narrative business report
    from the forecast & reorder planning table.
    """
    if client is None:
        return "OPENAI_API_KEY is not set. Unable to generate AI report."

    table_md = build_forecast_summary_table(df_results, horizon_days)

    system_prompt = """
You are an experienced retail and inventory optimization consultant.
You analyze demand forecasts, stock levels, and recommended order quantities for small shops and supermarkets.
Your job is to write a clear, professional business report that a shop owner can read and act on.
Use simple but professional language, no jargon, and be specific and actionable.
"""

    user_prompt = f"""
Shop name: {shop_name}
Forecast horizon: {horizon_days} days

Here is the forecast and stock summary table in markdown format:

{table_md}

Using this data, produce a detailed business report with the following structure:

1. Executive Summary
   - 2–3 short paragraphs summarizing the overall demand situation, risks, and opportunities.

2. Demand & Inventory Analysis
   - Discuss general sales trends based on the predicted demand.
   - Highlight which products are likely to be understocked (stockout risk).
   - Highlight which products appear stable or potentially overstocked.

3. Product-Level Recommendations
   - For each product, clearly state:
     - Whether it is at risk of stockout or is stable.
     - Whether to increase, maintain, or reduce current order quantities.
     - If Recommended Order > 0, explain why and what action to take.

4. Risk & Opportunity Assessment
   - Identify top 3 risks (e.g., missed sales due to stockouts, tied-up cash in slow-moving items).
   - Identify top 3 opportunities (e.g., upsell, bundle, seasonal pushes).

5. Recommended Action Plan (Next {horizon_days} Days)
   - Bullet list of concrete actions for the shop owner to take.
   - Include priorities (High / Medium / Low).

Write the report in a professional tone, but easy for a non-technical shop owner to understand.
Do not restate the raw table; refer to products by name and focus on insight and action.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # adjust to another current GPT model if desired
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.4,
    )

    report_text = response.choices[0].message.content
    return report_text


# --------------- MAIN APP LOGIC ---------------

st.write("📁 Upload your historical sales data (.csv or .xlsx):")
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = load_and_clean_data(uploaded_file)

        st.success(f"Data successfully loaded! {df['product_name'].nunique()} products detected.")
        st.write(f"📆 Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
        st.write(f"🧾 Total records: {len(df):,}")

        if df is not None and not df.empty:
            st.write("### 🔍 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # -------- MULTI-PRODUCT FORECAST & REORDER --------

            st.write(f"### 📦 Full Product Demand Forecast (Next {FORECAST_DAYS_MULTI} Days)")

            df_results = build_multi_product_forecast(df, horizon_days=FORECAST_DAYS_MULTI)

            if df_results.empty:
                st.warning(
                    f"Not enough history per product to run the {FORECAST_DAYS_MULTI}-day multi-product forecast. "
                    f"Each product needs at least {MIN_HISTORY_DAYS} days of data."
                )
            else:
                st.dataframe(df_results, use_container_width=True)

                st.write("### 🔁 Reorder Recommendations")

                assumed_stock = st.number_input(
                    "Enter assumed current stock per product (you can refine per-product stock later):",
                    min_value=0,
                    value=20,
                    step=1,
                )

                stock_col = "Current Stock"
                rec_col = "Recommended Order"
                risk_col = "Risk Status"
                demand_col = f"Predicted Demand Next {FORECAST_DAYS_MULTI} Days"

                df_results[stock_col] = assumed_stock
                df_results[rec_col] = np.maximum(
                    df_results[demand_col] - df_results[stock_col],
                    0,
                )

                def risk_label(order_qty: float) -> str:
                    if order_qty > 0:
                        return "⚠ Stockout Risk"
                    return "✔ Stable"

                df_results[risk_col] = df_results[rec_col].apply(risk_label)

                st.dataframe(df_results, use_container_width=True)

                # Excel export
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_results.to_excel(writer, index=False, sheet_name="Forecast & Reorder")

                st.download_button(
                    label="📥 Download Forecast & Reorder Report (Excel)",
                    data=buffer.getvalue(),
                    file_name="smartinventory_forecast_reorder.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # -------- AI BUSINESS REPORT (GPT) --------

                st.write("### 🤖 AI-Generated Business Report")

                if client is None:
                    st.info(
                        "To enable AI reports, set OPENAI_API_KEY in your Streamlit secrets."
                    )
                else:
                    shop_name = st.text_input("Shop name (for the report):", value="Demo Shop")
                    if st.button("🧠 Generate AI Inventory Report"):
                        with st.spinner("Generating AI-powered business report..."):
                            report_text = generate_ai_business_report(
                                df_results, shop_name=shop_name, horizon_days=FORECAST_DAYS_MULTI
                            )
                        st.markdown("#### 📄 Report")
                        st.markdown(report_text)

                        st.download_button(
                            label="📥 Download Report as Text",
                            data=report_text,
                            file_name="SmartInventory_AI_Report.txt",
                            mime="text/plain",
                        )

            st.write("---")

            # -------- SINGLE PRODUCT DETAILED FORECAST --------

            st.write("### 🔬 Detailed Forecast for a Single Product")

            product_list = sorted(df["product_name"].unique())
            selected_product = st.selectbox("Select product to forecast", product_list)

            forecast = generate_forecast(df, selected_product)

            if forecast is not None:
                st.subheader(f"Forecast for {selected_product}")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", linewidth=2)
                ax.fill_between(
                    forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    alpha=0.2,
                    label="Confidence Range",
                )
                ax.set_title(f"📈 Forecasted Demand for {selected_product}", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("Estimated Quantity Sold")
                ax.legend()
                st.pyplot(fig)

                csv_output = forecast[["ds", "yhat"]].copy()
                csv_output.rename(columns={"ds": "date", "yhat": "forecast_quantity"}, inplace=True)

                st.download_button(
                    label="Download Forecast CSV",
                    data=csv_output.to_csv(index=False),
                    file_name=f"{selected_product}_forecast.csv",
                    mime="text/csv",
                )
            else:
                st.warning(
                    f"Not enough data to generate forecast for {selected_product} "
                    f"(need at least {MIN_HISTORY_DAYS} days of history)."
                )

    except Exception as e:
        st.error(f"Error while processing file: {e}")

# --------------- FOOTER ---------------

st.write("---")
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
    🔹 SmartInventory AI | Demand Forecasting Assistant 🔹<br>
    Contact: smartinventory.ai@gmail.com
    </p>
    """,
    unsafe_allow_html=True,
)
