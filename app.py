import io
import pandas as pd
import streamlit as st
from prophet import Prophet

MIN_HISTORY_DAYS = 60
FORECAST_DAYS = 30

st.set_page_config(page_title="AI Demand Forecasting", layout="wide")
st.title("📈 AI Demand Forecasting for Small Shops")

st.write("Upload your sales history, and get product-level forecasts for the next 30 days.")

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

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    st.success("File uploaded successfully!")
    df = load_and_clean_data(uploaded_file)

    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df.head())

        product_list = sorted(df["product_name"].unique())
        selected_product = st.selectbox("Select product to forecast", product_list)

        forecast = generate_forecast(df, selected_product)

        if forecast is not None:
            st.subheader(f"Forecast for {selected_product}")
            st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))

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
