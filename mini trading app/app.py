import streamlit as st
import pandas as pd
import datetime
import simfin as sf
from simfin.names import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from joblib import load  # Import joblib
import transformation  # Import your transformation module
import logging
import io

# Configure logging
logging.basicConfig(level=logging.ERROR)  # Change to DEBUG for more detailed logs

# Set up SimFin API Key
SIMFIN_API_KEY = "163d13dc-55cd-4251-a5cf-137170c8dda2"  # Replace with your actual key
sf.set_api_key(SIMFIN_API_KEY)
sf.set_data_dir('simfin_data/')

# Load stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        df = sf.load_shareprices(market='us')

        if df is None:  # Check if df is None
            logging.error(f"Error: SimFin returned None for stock prices.")
            return None

        if ticker not in df.index.get_level_values(0).unique():
            st.error(f"⚠️ No stock data for {ticker}")
            return None

        stock_df = df.loc[ticker]
        stock_df = stock_df[(stock_df.index >= str(start_date)) & (stock_df.index <= str(end_date))]

        if stock_df.empty:
            st.warning(f"⚠️ No data in selected range.")
            return None

        return stock_df[['Close']]
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        st.error(f"Error fetching stock data: {e}")
        return None

@st.cache_data
def get_financial_statements(ticker):
    try:
        df = sf.load_income(variant='annual', market='us')
        if df is None:
            logging.error(f"Error: SimFin returned None for financial statements.")
            return None

        if ticker not in df.index.get_level_values(0).unique():
            st.error(f"⚠️ No financial data for {ticker}")
            return None

        return df.loc[ticker]
    except Exception as e:
        logging.error(f"Error fetching financials: {e}")
        st.error(f"Error fetching financials: {e}")
        return None

def remove_empty_columns(df):
    """Removes columns from a Pandas DataFrame where all values are missing."""
    try:
        if df is None or df.empty:
            return df  # Return None or empty DataFrame as is
        df_cleaned = df.dropna(axis=1, how='all')
        return df_cleaned
    except Exception as e:
        logging.error(f"Error removing empty columns: {e}")
        st.error(f"Error removing empty columns: {e}")
        return None # Keep the return, but return the original df

# Streamlit config
st.set_page_config(page_title="Automated Trading System", layout="wide")
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Go Live"])

# ================== HOME PAGE ===================
if page == "Home":
    st.title("📊 Automated Trading System")
    st.markdown("""
    Welcome to the **Automated Trading System** Use real-time data and machine learning to simulate and analyze stock investments.
    """)

    st.subheader("👨‍💻 Meet the Team")
    st.markdown("""
    - SAMEEN YOUNAS
     
    """)

    st.subheader("🎯 Purpose")
    st.markdown("Empower users to make better trading decisions using financial data and ML.")

# ================== GO LIVE PAGE ===================
elif page == "Go Live":
    st.title("🚀 Go Live - Real-Time Stock Analysis")

    # Define available companies
    companies = {
         "AAL": "American Airlines Group Inc.",
        "AAT": "American Assets Trust, Inc.",
        "ABNB": "Airbnb, Inc.",
        "AAPL": "Apple",
        "TSLA": "Tesla",
        "WMT": "Walmart",
        "CXW": "CoreCivic, Inc.",
        "META": "Facebook",
        "UNH": "UnitedHealth Group Inc",
        "COST": "Costco Wholesale"
    }

    # Sidebar stock selection with dropdown
    st.sidebar.header("📌 Stock Selection")
    selected_ticker = st.sidebar.selectbox(
        "📌 Select a Company",
        options=list(companies.keys()),
        format_func=lambda x: f"{x} - {companies[x]}"
    )
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2023, 1, 1))
    end_date = st.sidebar.date_input("📅 End Date", datetime.date.today())

    # Load data
    stock_data = get_stock_data(selected_ticker, start_date, end_date)
    financial_data = get_financial_statements(selected_ticker)
       # ================== STOCK PRICE CHART ===================
    if stock_data is not None and not stock_data.empty:
        stock_data = remove_empty_columns(stock_data)
        st.subheader(f"📈 Stock Price Trends for {selected_ticker}")
        stock_data.index = pd.to_datetime(stock_data.index)

        # Calculate Moving Averages
        stock_data["EMA_10"] = stock_data["Close"].ewm(span=10, adjust=False).mean()
        stock_data["EMA_26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data.index, stock_data['Close'], label="Closing Price", color="blue", linewidth=2)
        ax.plot(stock_data.index, stock_data["EMA_10"], label="EMA 10", color="green", linestyle="dashed", linewidth=1.5)
        ax.plot(stock_data.index, stock_data["EMA_26"], label="EMA 26", color="red", linestyle="dashed", linewidth=1.5)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price ($)")
        ax.set_title(f"{selected_ticker} Stock Price Trends")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

        st.dataframe(stock_data.sort_values(by='Close', ascending=False))
    else:
        st.warning("⚠️ No stock data for selected period.")
    # Show financials
    if financial_data is not None and not financial_data.empty:
        financial_data = remove_empty_columns(financial_data) # Clean financial data
        st.subheader(f"💰 {selected_ticker} Financial Statements (Annual)")
        st.dataframe(financial_data)
    else:
        st.warning("⚠️ No financial statements available.")

    # ================== ML PREDICTION ===================
    st.subheader("🔮 Market Movement Prediction")

    try:
        model = load("optimized_model.joblib")
        st.success("✅ ML Model loaded!")

        if stock_data is not None and not stock_data.empty:
            df_pred = stock_data.copy().reset_index()
            df_pred.rename(columns={'index': 'Date'}, inplace=True)
            df_pred['Date'] = pd.to_datetime(df_pred['Date'])
            df_pred['Ticker'] = selected_ticker
            df_pred['High'] = df_pred['Close']
            df_pred['Low'] = df_pred['Close']
            df_pred['Open'] = df_pred['Close']
            df_pred['Volume'] = 1000000  # placeholder
            df_pred['Adjusted Closing Price'] = df_pred['Close']  # ✅ FIXED

            # Corrected line: Call a function from the transformation module
            df_transformed = transformation.transformation(df_pred) # Assuming the function is named transformation
            if df_transformed is None or df_transformed.empty:
                st.warning("⚠️ No data after transformation.")
                
            df_transformed = remove_empty_columns(df_transformed) # Clean transformed data
            df_transformed = df_transformed.dropna()

            if not df_transformed.empty:
                latest_row = df_transformed.tail(1)
                features = ['Adjusted Closing Price', 'Low', 'Close', 'High', 'Open', 'EMA_10', 'EMA_26', 'Year']
                input_features = latest_row[features]

                predicted_close = model.predict(input_features)[0]
                last_close = latest_row['Close'].values[0]

                st.metric(label="📉 Last Close Price", value=f"${last_close:.2f}")
                st.metric(label="📈 Predicted Next-Day Close", value=f"${predicted_close:.2f}")

                if predicted_close > last_close:
                    st.success("📢 **Signal: BUY**")
                elif predicted_close < last_close:
                    st.error("📢 **Signal: SELL**")
                else:
                    st.info("📢 **Signal: HOLD**")
            else:
                st.warning("⚠️ No transformed data to predict.")

        else:
            st.warning("⚠️ No stock data to predict.")
    except Exception as e:
        logging.error(f"❌ Error loading model or making prediction: {e}")
        st.error(f"❌ Error loading model or making prediction: {e}")

    # ================== BACKTESTING ===================
    if st.checkbox("📊 Trading Strategy "):
        st.markdown("📅 Simulate past investments to see potential profits.")
        initial_investment = st.number_input("💵 Initial Investment ($):", min_value=1000, value=10000, step=500)
        start_simulation = st.button("Run Backtest")

        if start_simulation:
            if stock_data is not None and not stock_data.empty:
                prices = stock_data['Close']
                start_price = prices.iloc[0]
                end_price = prices.iloc[-1]

                shares_bought = initial_investment / start_price
                final_value = shares_bought * end_price
                profit = final_value - initial_investment
                return_pct = (profit / initial_investment) * 100

                st.metric(label="📈 Start Price", value=f"${start_price:.2f}")
                st.metric(label="🏁 End Price", value=f"${end_price:.2f}")
                st.metric(label="💰 Final Value", value=f"${final_value:.2f}")
                st.metric(label="📊 Return", value=f"{return_pct:.2f}%")

                if profit > 0:
                    st.success(f"🎉 Profit: ${profit:.2f}")
                else:
                    st.error(f"💸 Loss: ${abs(profit):.2f}")
            else:
                st.warning("⚠️ No stock data for backtesting.")

# ================== FOOTER ===================
st.markdown("---")
st.markdown("🚀 Built with SimFin API & Streamlit | Group 7 - Automated Trading System")