import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
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
    return df

# Load data
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

# Holders Over Time
st.header("Holders Over Time")
holders_over_time = df[df['balance'] > 0].groupby(
    df['timestamp'].dt.date)['address'].nunique().reset_index()
holders_over_time.columns = ['Date', 'Unique Holders']
fig_holders = px.line(
    holders_over_time, x='Date', y='Unique Holders',
    title='Number of Unique Holders Over Time')
st.plotly_chart(fig_holders, use_container_width=True)

# Adjusted Balance Distribution
st.header("Balance Distribution")
balance_bins = [0, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, np.inf]
balance_labels = ['<10k', '10k-50k', '50k-100k', '100k-500k', '500k-1M', '1M-5M', '>5M']
current_df['Balance Range'] = pd.cut(
    current_df['balance'], bins=balance_bins, labels=balance_labels)
balance_distribution = current_df['Balance Range'].value_counts().sort_index()
fig_balance_dist = px.bar(
    balance_distribution, x=balance_distribution.index.astype(str), y=balance_distribution.values,
    labels={'x': 'Balance Range', 'y': 'Number of Holders'},
    title='Distribution of Holder Balances')
st.plotly_chart(fig_balance_dist, use_container_width=True)

# Top Holders Analysis
st.header("Top Holders Analysis")
top_n = st.slider('Select Number of Top Holders to Display', min_value=10, max_value=100, value=20, step=10)
top_holders = current_df.nlargest(top_n, 'balance')

# Display Top Holders Table
st.subheader(f"Top {top_n} Holders")
st.dataframe(top_holders[['address', 'balance', 'unrealized_profit', 'num_xs_profit']])

# Top Holders Balances Chart
fig_top_holders = px.bar(
    top_holders.sort_values(by='balance', ascending=False),
    x='address', y='balance',
    labels={'address': 'Address', 'balance': 'Balance'},
    title=f'Top {top_n} Holders by Balance')
st.plotly_chart(fig_top_holders, use_container_width=True)

# Trading Activity of Top Holders
st.subheader(f"Trading Activity of Top {top_n} Holders")
activity_cols = [
    'quantity_bought', 'value_bought_btc', 'quantity_sold', 'value_sold_btc',
    'quantity_bought_1h', 'quantity_sold_1h',
    'quantity_bought_4h', 'quantity_sold_4h',
    'quantity_bought_24h', 'quantity_sold_24h',
    'quantity_bought_7d', 'quantity_sold_7d'
]
top_holders_activity = top_holders[['address'] + activity_cols]
st.dataframe(top_holders_activity)

# Price Levels of Buying and Selling
st.subheader("Price Levels of Buying and Selling by Top Holders")
# Calculate average price per unit in BTC
top_holders_activity['avg_buy_price_btc'] = top_holders_activity['value_bought_btc'] / top_holders_activity['quantity_bought']
top_holders_activity['avg_sell_price_btc'] = top_holders_activity['value_sold_btc'] / top_holders_activity['quantity_sold']

price_levels = top_holders_activity[['address', 'avg_buy_price_btc', 'avg_sell_price_btc']]
st.dataframe(price_levels)

# Plotting Average Buy and Sell Prices
fig_price_levels = go.Figure()
fig_price_levels.add_trace(go.Bar(
    x=price_levels['address'],
    y=price_levels['avg_buy_price_btc'],
    name='Avg Buy Price',
    marker_color='green'
))
fig_price_levels.add_trace(go.Bar(
    x=price_levels['address'],
    y=price_levels['avg_sell_price_btc'],
    name='Avg Sell Price',
    marker_color='red'
))
fig_price_levels.update_layout(
    barmode='group',
    xaxis_title='Address',
    yaxis_title='Price (BTC)',
    title='Average Buy and Sell Prices of Top Holders'
)
st.plotly_chart(fig_price_levels, use_container_width=True)

# Additional Metrics from Existing Metrics
st.header("Additional Metrics")
# Example: Profitability Ratio
current_df['profitability_ratio'] = current_df['unrealized_profit'] / current_df['balance']
profitability_df = current_df[['address', 'balance', 'unrealized_profit', 'profitability_ratio']]
st.dataframe(profitability_df.sort_values(by='profitability_ratio', ascending=False).head(20))

# Trading Volumes Over Time
st.header("Trading Volumes Over Time (All Holders)")
interval_options = ['1h', '4h', '24h', '7d']
selected_interval = st.selectbox('Select Time Interval for Trading Volumes', interval_options)

quantity_bought_col = f'quantity_bought_{selected_interval}'
quantity_sold_col = f'quantity_sold_{selected_interval}'

# Aggregate trading volumes over time
trading_volumes_over_time = df.groupby(df['timestamp'].dt.date).agg({
    quantity_bought_col: 'sum',
    quantity_sold_col: 'sum'
}).reset_index().fillna(0)

# Plot trading volumes
fig_trading_volumes = go.Figure()
fig_trading_volumes.add_trace(go.Bar(
    x=trading_volumes_over_time['timestamp'], y=trading_volumes_over_time[quantity_bought_col],
    name='Quantity Bought', marker_color='green'))
fig_trading_volumes.add_trace(go.Bar(
    x=trading_volumes_over_time['timestamp'], y=trading_volumes_over_time[quantity_sold_col],
    name='Quantity Sold', marker_color='red'))
fig_trading_volumes.update_layout(
    barmode='group', xaxis_title='Date', yaxis_title='Quantity',
    title=f'Trading Volumes Over Time ({selected_interval} Interval)')
st.plotly_chart(fig_trading_volumes, use_container_width=True)

# Footer
st.markdown("""
---
*Note: Data is updated every minute. Latest data timestamp is displayed above.*
""")
