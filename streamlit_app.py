import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Function to load data from the PostgreSQL database and handle NULL values
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
    
    # Correctly parse timestamps with timezone information
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Replace [NULL] or missing values
    df.fillna({
        'quantity_bought': 0, 'value_bought_btc': 0,
        'quantity_bought_1h': 0, 'value_bought_1h_btc': 0,
        'quantity_bought_4h': 0, 'value_bought_4h_btc': 0,
        'quantity_bought_24h': 0, 'value_bought_24h_btc': 0,
        'quantity_bought_7d': 0, 'value_bought_7d_btc': 0,
        'quantity_sold': 0, 'avg_quantity_sold': 0,
        'value_sold_btc': 0, 'quantity_sold_1h': 0,
        'value_sold_1h_btc': 0, 'quantity_sold_4h': 0,
        'value_sold_4h_btc': 0, 'quantity_sold_24h': 0,
        'value_sold_24h_btc': 0, 'quantity_sold_7d': 0,
        'value_sold_7d_btc': 0
    }, inplace=True)

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
col3.metric("Total Unrealized Profit", f"{total_unrealized_profit:,.6f} BTC")
col4.metric("Average Profit Multiplier", f"{average_num_xs_profit:.2f}x")

# Buyers and Sellers Over Time Chart
st.header("Unique Buyers and Sellers Over Time")
buyers_sellers_over_time = df.groupby(df['timestamp'].dt.floor('T')).agg({
    'quantity_bought': lambda x: x.notnull().sum(),
    'quantity_sold': lambda x: x.notnull().sum()
}).reset_index()

fig_buyers_sellers = go.Figure()
fig_buyers_sellers.add_trace(go.Scatter(
    x=buyers_sellers_over_time['timestamp'], y=buyers_sellers_over_time['quantity_bought'],
    mode='lines', name='Unique Buyers'))
fig_buyers_sellers.add_trace(go.Scatter(
    x=buyers_sellers_over_time['timestamp'], y=buyers_sellers_over_time['quantity_sold'],
    mode='lines', name='Unique Sellers'))
fig_buyers_sellers.update_layout(
    xaxis_title='Timestamp', yaxis_title='Number of Unique Buyers/Sellers',
    title='Unique Buyers and Sellers Over Time')
st.plotly_chart(fig_buyers_sellers, use_container_width=True)

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

# Top Holders Recent Buys and Sells by Interval
st.header("Top Holders' Recent Buys and Sells by Time Intervals")
interval_options = ['1h', '4h', '24h', '7d']
top_n = st.slider('Select Number of Top Holders to Display', min_value=10, max_value=100, value=20, step=10)
selected_interval = st.selectbox('Select Time Interval for Analysis', interval_options)

quantity_bought_col = f'quantity_bought_{selected_interval}'
quantity_sold_col = f'quantity_sold_{selected_interval}'
value_bought_col = f'value_bought_{selected_interval}'
value_sold_col = f'value_sold_{selected_interval}'

# Check if the columns exist; if not, initialize them to 0 to avoid errors
for col in [quantity_bought_col, quantity_sold_col, value_bought_col, value_sold_col]:
    if col not in df.columns:
        df[col] = 0

top_holders = current_df.nlargest(top_n, 'balance')

# Display Top Holders Table
st.subheader(f"Top {top_n} Holders Recent Buys and Sells")
st.dataframe(top_holders[['address', 'balance', quantity_bought_col, value_bought_col, quantity_sold_col, value_sold_col]])

# Plot Buys and Sells for Top Holders by Interval
fig_top_holders_buys_sells = go.Figure()
fig_top_holders_buys_sells.add_trace(go.Bar(
    x=top_holders['address'], y=top_holders[quantity_bought_col],
    name='Buys', marker_color='green'))
fig_top_holders_buys_sells.add_trace(go.Bar(
    x=top_holders['address'], y=top_holders[quantity_sold_col],
    name='Sells', marker_color='red'))
fig_top_holders_buys_sells.update_layout(
    barmode='group', xaxis_title='Address', yaxis_title='Quantity',
    title=f'Top {top_n} Holders - Buys and Sells ({selected_interval} Interval)',
    xaxis_tickangle=-45)
st.plotly_chart(fig_top_holders_buys_sells, use_container_width=True)

# Summary Table for Total Buys and Sells by Time Interval
st.header(f"Total Buys and Sells Over {selected_interval} Interval")
total_summary = current_df.groupby(df['timestamp'].dt.floor('T')).agg({
    quantity_bought_col: 'sum',
    quantity_sold_col: 'sum'
}).reset_index()

st.subheader(f"Total Buys and Sells Over {selected_interval}")
st.dataframe(total_summary)

# Buys Volume Over Time (Changed from Buys and Sells)
st.header("Buy Volume Over Time")
buy_volume_over_time = df.groupby(df['timestamp'].dt.floor('T'))[quantity_bought_col].sum().reset_index()

fig_buy_volume = px.bar(
    buy_volume_over_time, x='timestamp', y=quantity_bought_col,
    title=f'Buy Volume Over Time ({selected_interval} Interval)',
    labels={quantity_bought_col: 'Buy Volume', 'timestamp': 'Timestamp'})
fig_buy_volume.update_layout(xaxis_title='Timestamp', yaxis_title='Buy Volume')
st.plotly_chart(fig_buy_volume, use_container_width=True)

# Footer
st.markdown("""
---
*Note: Data is updated every minute. Latest data timestamp is displayed above.*
""")
