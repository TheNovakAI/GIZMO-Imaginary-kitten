import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime

# Function to load data from the PostgreSQL database
@st.cache_data(ttl=600)
def load_data():
    # Retrieve database credentials from Streamlit secrets
    db_credentials = st.secrets["postgres"]
    connection_string = (
        f"postgresql://{db_credentials['user']}:{db_credentials['password']}"
        f"@{db_credentials['host']}:{db_credentials['port']}/{db_credentials['dbname']}"
    )
    engine = create_engine(connection_string)
    query = """
    SELECT * FROM gizmo_holders_balances_history;
    """
    df = pd.read_sql_query(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Load data
st.title("Gizmo Meme Coin Dashboard")
st.markdown("""
Welcome to the Gizmo Meme Coin Dashboard! Explore trading metrics, holders' data, and market trends to make informed trading decisions.
""")

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('')

# Sidebar filters
st.sidebar.header("Filters")
unique_addresses = df['address'].unique()
selected_addresses = st.sidebar.multiselect(
    'Select Addresses (Leave empty to select all)', unique_addresses)

start_date = st.sidebar.date_input(
    "Start Date", value=df['timestamp'].min().date())
end_date = st.sidebar.date_input(
    "End Date", value=df['timestamp'].max().date())

if start_date > end_date:
    st.sidebar.error("Error: End date must fall after start date.")

# Apply filters
filtered_df = df.copy()
if selected_addresses:
    filtered_df = filtered_df[filtered_df['address'].isin(selected_addresses)]
filtered_df = filtered_df[
    (filtered_df['timestamp'].dt.date >= start_date) &
    (filtered_df['timestamp'].dt.date <= end_date)
]

# KPI Metrics
st.header("Key Performance Indicators (KPIs)")
total_holders = filtered_df['address'].nunique()
total_balance = filtered_df['balance'].sum()
total_unrealized_profit = filtered_df['unrealized_profit'].sum()
average_num_xs_profit = filtered_df['num_xs_profit'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Holders", f"{total_holders}")
col2.metric("Total Balance", f"{total_balance:,.2f}")
col3.metric("Total Unrealized Profit", f"{total_unrealized_profit:,.2f} BTC")
col4.metric("Average Profit Multiplier", f"{average_num_xs_profit:.2f}x")

# Holders Over Time
st.header("Holders Over Time")
holders_over_time = filtered_df.groupby(
    filtered_df['timestamp'].dt.date)['address'].nunique().reset_index()
holders_over_time.columns = ['Date', 'Unique Holders']
fig_holders = px.line(
    holders_over_time, x='Date', y='Unique Holders',
    title='Number of Unique Holders Over Time')
st.plotly_chart(fig_holders, use_container_width=True)

# Balance Distribution
st.header("Balance Distribution")
balance_bins = [0, 1000, 5000, 10000, 50000, 100000, np.inf]
balance_labels = ['<1k', '1k-5k', '5k-10k', '10k-50k', '50k-100k', '>100k']
filtered_df['Balance Range'] = pd.cut(
    filtered_df['balance'], bins=balance_bins, labels=balance_labels)
balance_distribution = filtered_df['Balance Range'].value_counts().sort_index()
fig_balance_dist = px.bar(
    balance_distribution, x=balance_distribution.index, y=balance_distribution.values,
    labels={'x': 'Balance Range', 'y': 'Number of Holders'},
    title='Distribution of Holder Balances')
st.plotly_chart(fig_balance_dist, use_container_width=True)

# Top Holders
st.header("Top Holders")
top_n = st.slider('Select Number of Top Holders to Display', min_value=5, max_value=50, value=10)
top_holders = filtered_df.groupby('address')['balance'].max().nlargest(top_n).reset_index()
fig_top_holders = px.bar(
    top_holders, x='address', y='balance',
    labels={'address': 'Address', 'balance': 'Balance'},
    title=f'Top {top_n} Holders by Balance')
st.plotly_chart(fig_top_holders, use_container_width=True)

# Trading Volumes
st.header("Trading Volumes Over Time")
interval_options = ['1h', '4h', '24h', '7d']
selected_interval = st.selectbox('Select Time Interval', interval_options)

quantity_bought_col = f'quantity_bought_{selected_interval}'
value_bought_col = f'value_bought_{selected_interval}_btc'
quantity_sold_col = f'quantity_sold_{selected_interval}'
value_sold_col = f'value_sold_{selected_interval}_btc'

# Aggregate trading volumes
trading_volumes = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
    quantity_bought_col: 'sum',
    value_bought_col: 'sum',
    quantity_sold_col: 'sum',
    value_sold_col: 'sum'
}).reset_index().fillna(0)

# Plot trading volumes
fig_trading_volumes = go.Figure()
fig_trading_volumes.add_trace(go.Bar(
    x=trading_volumes['timestamp'], y=trading_volumes[quantity_bought_col],
    name='Quantity Bought', marker_color='green'))
fig_trading_volumes.add_trace(go.Bar(
    x=trading_volumes['timestamp'], y=trading_volumes[quantity_sold_col],
    name='Quantity Sold', marker_color='red'))
fig_trading_volumes.update_layout(
    barmode='group', xaxis_title='Date', yaxis_title='Quantity',
    title=f'Trading Volumes ({selected_interval} Interval)')
st.plotly_chart(fig_trading_volumes, use_container_width=True)

# Profitability Analysis
st.header("Profitability Analysis")
fig_profitability = px.scatter(
    filtered_df, x='num_xs_profit', y='unrealized_profit',
    size='balance', color='balance',
    hover_data=['address'],
    labels={'num_xs_profit': 'Profit Multiplier (X)', 'unrealized_profit': 'Unrealized Profit (BTC)'},
    title='Profit Multiplier vs. Unrealized Profit')
st.plotly_chart(fig_profitability, use_container_width=True)

# Data Table
st.header("Detailed Data Table")
st.dataframe(filtered_df.sort_values(by='unrealized_profit', ascending=False).reset_index(drop=True))

# Download Data
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df(filtered_df)
st.download_button(
    label="Download Data as CSV",
    data=csv_data,
    file_name='gizmo_coin_data.csv',
    mime='text/csv',
)

# Footer
st.markdown("""
---
*Note: Data is updated every 10 minutes.*
""")
