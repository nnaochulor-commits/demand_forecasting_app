import io
import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np


MIN_HISTORY_DAYS = 60
FORECAST_DAYS = 30


st.set_page_config(page_title="LyZeR AI", layout="wide")

st.markdown(
    """
    <style>
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
    .css-18e3th9 {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True
)


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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">LyZER AI — Demand Forecasting Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload your historical sales data and instantly get AI-powered demand forecasts and reorder suggestions.</p>', unsafe_allow_html=True)

st.write("---")

st.sidebar.title("📘 Quick Guide")
st.sidebar.info(
    """
    1️⃣ Upload your sales data  
    2️⃣ Select a product to forecast  
    3️⃣ View demand prediction for next 30 days  
    4️⃣ Download forecast report  
    """
)
st.sidebar.write("💬 Contact for custom model:")
st.sidebar.write("smartinventory.ai@gmail.com")


def load_and_clean_data(file):
    try:
        df = pd.read_csv(file)
    except:
        file.seek(0)
        df = pd.read_excel(file)

    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "product_name", "quantity"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        st.error(f"Missing required columns: {missing}")
        return None
    
    df["date"] = pd.to_datetime(df["date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    
    df = df.groupby(["product_name", "date"], as_index=False)["quantity"].sum()
    df.rename(columns={"date": "ds", "quantity": "y"}, inplace=True)
    
    return df

def generate_forecast(df, product):
    product_df = df[df["product_name"] == product]
    if len(product_df) < MIN_HISTORY_DAYS:
        return None

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(product_df)
    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    return forecast

uploaded_file = st.file_uploader("📁 Upload your historical sales data (.csv or .xlsx):", type=["csv", "xlsx"])
st.info("💡 Tip: Export past sales from your POS or Excel. You only need Date, Product Name, and Quantity.")


if uploaded_file:
    df = load_and_clean_data(uploaded_file)

    st.success(f"Data successfully loaded! {df['product_name'].nunique()} products detected.")
    st.write(f"📆 Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
    st.write(f"🧾 Total records: {len(df):,}")


    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df.head())

        st.write("### 📦 Full Product Demand Forecast (Next 14 Days)")

        full_forecast_results = []

        for product in df['product_name'].unique():
            product_df = df[df['product_name'] == product][['ds', 'y']].copy()
            
            try:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model.fit(product_df)
                future = model.make_future_dataframe(periods=14)
                forecast = model.predict(future)
                
                total_forecast = forecast.tail(14)['yhat'].sum()
                full_forecast_results.append([product, round(total_forecast, 1)])
            except:
                continue

        df_results = pd.DataFrame(full_forecast_results, columns=['Product', 'Predicted Demand Next 14 Days'])
        st.dataframe(df_results, use_container_width=True)

        # 🔁 Reorder recommendation section
        st.write("### 🔁 Reorder Recommendations (Next 14 Days)")

        if not df_results.empty:
            # Simple approach: one assumed current stock level for all products (MVP)
            assumed_stock = st.number_input(
                "Enter assumed current stock per product (you can refine per-product logic later):",
                min_value=0,
                value=20,
                step=1
                )

            # Add current stock and recommended order columns
            df_results["Current Stock"] = assumed_stock

            df_results["Recommended Order"] = np.maximum(
                df_results["Predicted Demand Next 14 Days"] - df_results["Current Stock"],
                0
            )

            # Risk status label
            def risk_label(order):
                if order > 0:
                    return "⚠ Stockout Risk"
                return "✔ Stable"

            df_results["Risk Status"] = df_results["Recommended Order"].apply(risk_label)

            st.dataframe(df_results, use_container_width=True)

            # Optional: allow download of this as an Excel report
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_results.to_excel(writer, index=False, sheet_name="Forecast & Reorder")

            st.download_button(
                label="📥 Download Forecast & Reorder Report (Excel)",
                data=buffer.getvalue(),
                file_name="smartinventory_forecast_reorder.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("No forecast results were generated. Check that you have enough history per product.")


        st.write("---")  # Optional separator

        product_list = sorted(df["product_name"].unique())
        selected_product = st.selectbox("Select product to forecast", product_list)

        forecast = generate_forecast(df, selected_product)

        if forecast is not None:
            st.subheader(f"Forecast for {selected_product}")
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", linewidth=2)
            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="Confidence Range")
            ax.set_title(f"📈 Forecasted Demand for {selected_product}", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Estimated Quantity Sold")
            ax.legend()
            st.pyplot(fig)

            csv_output = forecast[["ds", "yhat"]]
            csv_output.rename(columns={"ds": "date", "yhat": "forecast_quantity"}, inplace=True)

            st.download_button(
                label="Download Forecast CSV",
                data=csv_output.to_csv(index=False),
                file_name=f"{selected_product}_forecast.csv",
                mime="text/csv"
            )
        else:
            st.warning("Not enough data to generate forecast (need at least 60 days).")

st.write("---")
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
    🔹 SmartInventory AI | Demand Forecasting Assistant 🔹<br>
    Contact: smartinventory.ai@gmail.com
    </p>
    """,
    unsafe_allow_html=True
)

