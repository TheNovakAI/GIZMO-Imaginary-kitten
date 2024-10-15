import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
import numpy as np

# Function to load data from the PostgreSQL database
@st.cache_data(ttl=60)
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
    df.fillna(0, inplace=True)  # Replace null values with 0
    return df

# Page config for dark mode and wide layout
st.set_page_config(page_title="Gizmo Meme Coin Dashboard", layout="wide", page_icon=":rocket:", initial_sidebar_state="expanded")

# Load data into cache
st.title("Gizmo Meme Coin Dashboard")
st.markdown("""
Welcome to the Gizmo Meme Coin Dashboard! Explore trading metrics, holders' data, and market trends to make informed trading decisions.
""")

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('')

# Use the latest timestamp for current data
latest_timestamp = df['timestamp'].max()
current_df = df[df['timestamp'] == latest_timestamp].copy()
st.markdown(f"**Data as of:** {latest_timestamp}")

# Sidebar filters
st.sidebar.header("Filters")

# Default number of top holders to "all" holders
top_holders_limit = st.sidebar.number_input(
    'Enter the number of top holders (default to all)', min_value=1, max_value=len(current_df), value=len(current_df), step=1)

# Filter unique addresses
unique_addresses = current_df['address'].unique()
selected_addresses = st.sidebar.multiselect(
    'Select Addresses (Leave empty to select all)', unique_addresses)

if selected_addresses:
    current_df = current_df[current_df['address'].isin(selected_addresses)]

# KPI Metrics
st.header("Key Performance Indicators (KPIs)")
total_holders = current_df[current_df['balance'] > 0]['address'].nunique()
total_balance = current_df['balance'].sum()
total_unrealized_profit = current_df['unrealized_profit'].sum()
average_num_xs_profit = current_df['num_xs_profit'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Holders", f"{total_holders}")
col2.metric("Total Balance", f"{total_balance:,.2f}")
col3.metric("Total Unrealized Profit", f"{total_unrealized_profit:,.2f} BTC")
col4.metric("Average Profit Multiplier", f"{average_num_xs_profit:.2f}x")

# Holders Over Time - Ensure all timestamps are shown
st.header("Holders Over Time")
holders_over_time = df[df['balance'] > 0].groupby('timestamp')['address'].nunique().reset_index()
holders_over_time.columns = ['Timestamp', 'Unique Holders']

fig_holders = go.Figure()
fig_holders.add_trace(go.Scatter(x=holders_over_time['Timestamp'], y=holders_over_time['Unique Holders'], mode='lines+markers', name='Unique Holders'))
fig_holders.update_layout(title='Number of Unique Holders Over Time', xaxis_title='Timestamp', yaxis_title='Number of Holders')
st.plotly_chart(fig_holders, use_container_width=True)

# Adjusted Balance Distribution
st.header("Balance Distribution")
balance_bins = [0, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, np.inf]
balance_labels = ['<10k', '10k-50k', '50k-100k', '100k-500k', '500k-1M', '1M-5M', '>5M']
current_df['Balance Range'] = pd.cut(
    current_df['balance'], bins=balance_bins, labels=balance_labels)
balance_distribution = current_df['Balance Range'].value_counts().sort_index()
fig_balance_dist = go.Figure()
fig_balance_dist.add_trace(go.Bar(x=balance_distribution.index.astype(str), y=balance_distribution.values, name='Holders'))
fig_balance_dist.update_layout(title='Distribution of Holder Balances', xaxis_title='Balance Range', yaxis_title='Number of Holders')
st.plotly_chart(fig_balance_dist, use_container_width=True)

# Top Holders Trading Activity
st.header("Top Holders Trading Activity")
top_holders = current_df.nlargest(top_holders_limit, 'balance')

# Split top holders into separate Buy and Sell tables
st.subheader(f"Top {top_holders_limit} Buyers and Sellers")
interval_options = ['1h', '4h', '24h', '7d']
selected_interval = st.selectbox('Select Time Interval for Buy/Sell Activity', interval_options)

quantity_bought_col = f'quantity_bought_{selected_interval}'
quantity_sold_col = f'quantity_sold_{selected_interval}'

top_buyers = top_holders[top_holders[quantity_bought_col] > 0].sort_values(by=quantity_bought_col, ascending=False)
top_sellers = top_holders[top_holders[quantity_sold_col] > 0].sort_values(by=quantity_sold_col, ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Top Buyers in {selected_interval}")
    st.dataframe(top_buyers[['address', quantity_bought_col, 'value_bought_btc']])

with col2:
    st.subheader(f"Top Sellers in {selected_interval}")
    st.dataframe(top_sellers[['address', quantity_sold_col, 'value_sold_btc']])

# Unique Buyers and Sellers Over Time (non-cumulative, all timestamps shown)
st.header("Unique Buyers and Sellers Over Time")
buyers_sellers_over_time = df.groupby('timestamp').agg({
    'quantity_bought': 'nunique',
    'quantity_sold': 'nunique'
}).reset_index()
buyers_sellers_over_time.columns = ['Timestamp', 'Unique Buyers', 'Unique Sellers']

# Create bar chart to compare unique buyers and sellers
fig_buyers_sellers = go.Figure()
fig_buyers_sellers.add_trace(go.Bar(
    x=buyers_sellers_over_time['Timestamp'], y=buyers_sellers_over_time['Unique Buyers'],
    name='Unique Buyers', marker_color='green'))
fig_buyers_sellers.add_trace(go.Bar(
    x=buyers_sellers_over_time['Timestamp'], y=buyers_sellers_over_time['Unique Sellers'],
    name='Unique Sellers', marker_color='red'))

fig_buyers_sellers.update_layout(
    barmode='group',  # Bars will be side-by-side for comparison
    xaxis_title='Timestamp', yaxis_title='Count',
    title='Unique Buyers and Sellers Per Time Interval')
st.plotly_chart(fig_buyers_sellers, use_container_width=True)

# Price Over Time and Unrealized Profit Comparison
st.header("Unrealized Profit vs. Average Price Over Time (in Sats)")

# Calculate average price as sats per token (using buys only)
df['avg_price_bought_sats'] = (df['value_bought_4h_btc'] / df['quantity_bought_4h']) / 0.00000001

price_vs_profit = df.groupby(df['timestamp'].dt.floor('10min')).agg({
    'avg_price_bought_sats': 'mean',
    'unrealized_profit': 'sum'
}).reset_index()

fig_price_vs_profit = go.Figure()

# Unrealized profit line
fig_price_vs_profit.add_trace(go.Scatter(
    x=price_vs_profit['timestamp'], y=price_vs_profit['unrealized_profit'],
    mode='lines', name='Unrealized Profit (BTC)', yaxis='y1', line=dict(color='blue')))

# Avg price bought in sats line
fig_price_vs_profit.add_trace(go.Scatter(
    x=price_vs_profit['timestamp'], y=price_vs_profit['avg_price_bought_sats'],
    mode='lines', name='Avg Price Bought (Sats)', yaxis='y2', line=dict(color='orange')))

# Layout for dual axis
fig_price_vs_profit.update_layout(
    title='Unrealized Profit vs. Avg Price Bought Over Time (Sats)',
    xaxis=dict(title='Timestamp'),
    yaxis=dict(title='Unrealized Profit (BTC)', side='left'),
    yaxis2=dict(title='Avg Price Bought (Sats)', overlaying='y', side='right'),
    legend=dict(x=0.1, y=1.1)
)

st.plotly_chart(fig_price_vs_profit, use_container_width=True)

# Footer
st.markdown("""
---
*Note: Data is updated every minute. Latest data timestamp is displayed above.*
""")
