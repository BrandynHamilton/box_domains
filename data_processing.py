# %% [markdown]
# ***Can only retrieve up to 1000 data points from Flipside API***

# %%
import pandas as pd
import numpy as np 
import requests
import json
import time
from flipside import Flipside
import os
import traceback
from dotenv import load_dotenv
import datetime as dt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
#from prophet import Prophet
from dash import Dash, html, dcc, Input, Output, State, callback
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dash_table



from sql.sql_scripts import mints_query, sales_query, eth_price_query

# %%
load_dotenv()

# %%
pd.options.display.float_format = '{:,.2f}'.format


# %%
opensea_api_key = os.getenv('opensea_api_key')

# %% [markdown]
# ***Listing Data***

# %%
def fetch_listings(api_key, delay_between_requests=1):
    base_url = "https://api.opensea.io/api/v2/listings/collection/3dns-powered-domains/all"
    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }
    params = {"limit": 100} 

    listings = []
    page_count = 0

    while True:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            fetched_listings = data.get("listings", [])
            listings.extend(fetched_listings)
            page_count += 1
            
            # Extract and print the cursor
            next_cursor = data.get("next")
            print(f"Page {page_count}, Cursor: {next_cursor}, Listings Fetched: {len(fetched_listings)}")
            
            if next_cursor:
                params['next'] = next_cursor  # Update the 'next' parameter for the next request
            else:
                break  # No more pages to fetch
                
            # Implementing delay
            time.sleep(delay_between_requests)
            
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    print(f"Total pages fetched: {page_count}")
    print(f"Total listings fetched: {len(listings)}")
    
    df = pd.DataFrame(listings)
    return df

# %% [markdown]
# ***Descriptions***

# %%
def save_last_identifier(identifier):
    with open("last_identifier.txt", "w") as file:
        file.write(identifier)

def load_last_identifier():
    try:
        with open("last_identifier.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

# %%
def fetch_all_descriptions(api_key, delay_between_requests=1):
    base_url = "https://api.opensea.io/api/v2/collection/3dns-powered-domains/nfts"
    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }
    params = {"limit": 200}

    all_descriptions = []

    page_count = 0
    last_identifier = load_last_identifier()

    while True:
        if last_identifier:
            params['last_identifier'] = last_identifier

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            fetched_descriptions = data.get("nfts", [])
            
            if not fetched_descriptions:
                break

            # Process only name and identifier for each description
            for description in fetched_descriptions:
                processed_description = {
                    "name": description.get('name'),
                    "identifier": description.get('identifier')
                }
                all_descriptions.append(processed_description)
            
            # Update the last_identifier to the latest one fetched
            last_identifier = fetched_descriptions[-1].get('identifier')
            save_last_identifier(last_identifier)
            
            page_count += 1
            next_cursor = data.get("next")
            print(f"Page {page_count}, Cursor: {next_cursor} Descriptions Fetched: {len(fetched_descriptions)}, total fetched: {len(all_descriptions)}")
            
            if next_cursor:
                params['next'] = next_cursor
            else:
                break  # No more pages to fetch

            time.sleep(delay_between_requests)
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    print(f"Total pages fetched: {page_count}, Total descriptions fetched: {len(all_descriptions)}")
    
    # Save the processed descriptions to a file
    df = pd.DataFrame(all_descriptions)
    return df

# %% [markdown]
# ***Events***

# %%
import json
import os

def save_last_timestamp(event_type, timestamp):
    data = {}
    if os.path.exists("last_timestamps.json"):
        with open("last_timestamps.json", "r") as file:
            data = json.load(file)
    data[event_type] = timestamp
    with open("last_timestamps.json", "w") as file:
        json.dump(data, file)

def load_last_timestamp(event_type):
    if os.path.exists("last_timestamps.json"):
        with open("last_timestamps.json", "r") as file:
            data = json.load(file)
        return data.get(event_type, None)
    return None

# %%
def fetch_event_type(api_key, event_type, all_events, params, headers):
    base_url = f"https://api.opensea.io/api/v2/events/collection/3dns-powered-domains"
    params['event_type'] = event_type
    
    # Load the last timestamp/identifier
    last_timestamp = load_last_timestamp(event_type)
    if last_timestamp:
        params['occurred_after'] = last_timestamp
    
    page_count = 0
    while True:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            fetched_events = data.get("asset_events", [])
            all_events.extend(fetched_events)
            
            if fetched_events:
                # Update the last timestamp/identifier to the latest one fetched
                last_event_time = fetched_events[-1].get("created_date")
                save_last_timestamp(event_type, last_event_time)
            
            page_count += 1
            next_cursor = data.get("next")
            print(f"Fetching {event_type}: Page {page_count}, Events Fetched: {len(fetched_events)}, Total Events: {len(all_events)}, next cursor: {next_cursor}")
            
            if next_cursor:
                params['next'] = next_cursor
            else:
                break  # No more pages to fetch

            time.sleep(1)  # Delay between requests
        else:
            print(f"Failed to fetch {event_type} data: HTTP {response.status_code}, Response: {response.text}")
            break

def fetch_all_events(api_key):
    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }
    params = {
        "limit": 50  # Adjust the limit as needed
    }

    all_events = []

    # Fetch listings
    fetch_event_type(api_key, "listing", all_events, params.copy(), headers)

    # Fetch sales
    fetch_event_type(api_key, "sale", all_events, params.copy(), headers)

    # Save the fetched events to a DataFrame
    print(f"Total events fetched: {len(all_events)}")
    df = pd.DataFrame(all_events)
    return df 

# %% [markdown]
# ***Flipside Data***

# %%
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
flipside = Flipside(flipside_api_key, "https://api-v2.flipsidecrypto.xyz")

# %%
def flipside_api_results(query):
  query_result_set = flipside.query(query)
  # what page are we starting on?
  current_page_number = 1

  # How many records do we want to return in the page?
  page_size = 1000

  # set total pages to 1 higher than the `current_page_number` until
  # we receive the total pages from `get_query_results` given the 
  # provided `page_size` (total_pages is dynamically determined by the API 
  # based on the `page_size` you provide)

  total_pages = 2


  # we'll store all the page results in `all_rows`
  all_rows = []

  while current_page_number <= total_pages:
    results = flipside.get_query_results(
      query_result_set.query_id,
      page_number=current_page_number,
      page_size=page_size
    )

    total_pages = results.page.totalPages
    if results.records:
        all_rows = all_rows + results.records
    
    current_page_number += 1

  return pd.DataFrame(all_rows)

# %% [markdown]
# ***Data Retrieval/Processing***

# %%

def data_processing():
    mint_df = flipside_api_results(mints_query)
    mint_df.to_csv('data/mint_data.csv')

    # %% [markdown]
    # mint_df = pd.read_csv('data/mint_data.csv')
    # mint_df

    # %%
    sales_df = flipside_api_results(sales_query)
    sales_df.to_csv('data/sales_data.csv')


    # %% [markdown]
    # sales_df = pd.read_csv('data/sales_data.csv')
    # sales_df

    # %%
    eth_usd_df = flipside_api_results(eth_price_query)
    eth_usd_df.to_csv('data/eth_usd.csv')


    # %% [markdown]
    # eth_usd_df = pd.read_csv('data/eth_usd.csv')
    # eth_usd_df

    # %% [markdown]
    events_df = fetch_all_events(api_key= opensea_api_key)
    # events_df.to_json('data/events_data.json', orient='records', date_format='iso')
    # 

    # %%
    # events_df = pd.read_json('data/events_data.json', orient='records')
    events_df = events_df.dropna()

    # %% [markdown]
    descriptions_df = fetch_all_descriptions(api_key= opensea_api_key)
    # descriptions_df.to_json('data/descriptions_data.json', orient='records', date_format='iso')

    # %%
    # descriptions_df = pd.read_json('data/descriptions_data.json', orient='records')
    descriptions_df = descriptions_df.dropna()
    descriptions_df['name'].tail(20)

    # %% [markdown]
    listings_df = fetch_listings(api_key= opensea_api_key, delay_between_requests=1)
    # listings_df.to_json('data/listings_data.json', orient='records', date_format='iso')

    # %%
    # listings_df = pd.read_json('data/listings_data.json', orient='records')
    listings_df = listings_df.dropna()
    listings_df

    # %%
    def unpack_protocol_data(row):
        protocol_data = row['protocol_data']
        parameters = protocol_data.get('parameters', {})
        consideration = parameters.get('consideration', [{}])
        offer = parameters.get('offer', [{}])
        price = row['price']['current']
        
        chain = row['chain']
        order_hash = row['order_hash']
        currency = price.get('currency')
        price_string = price.get('value')
        price_in_eth = float(price_string) / (10 ** price.get('decimals', 18))
        primary_recipient = consideration[0].get('recipient') if consideration else None
        identifier_or_criteria = offer[0].get('identifierOrCriteria') if offer else None
        start_time = parameters.get('startTime')
        end_time = parameters.get('endTime')
        
        return pd.Series([
            chain, order_hash, currency, price_string, price_in_eth, 
            primary_recipient, identifier_or_criteria, start_time, end_time
        ])

    # %%
    unpacked_columns = listings_df.apply(unpack_protocol_data, axis=1)
    unpacked_columns.columns = [
        'chain', 'order_hash', 'currency', 'price_string', 'price_in_eth', 
        'primary_recipient', 'identifier_or_criteria', 'start_time', 'end_time'
    ]
    listings_df = pd.concat([listings_df, unpacked_columns], axis=1)
    listings_df = listings_df.drop(columns=['protocol_data'])


    # %%
    listings_df['identifier_or_criteria'] = listings_df['identifier_or_criteria'].astype(float)
    listings_df.rename(columns={'identifier_or_criteria':'tokenid'}, inplace=True)

    # %%
    listings_df.drop(columns=['price_string','currency','primary_recipient','chain','price','protocol_address','type'], inplace=True)

    # %%
    descriptions_df.rename(columns={'identifier':'tokenid'}, inplace=True)

    # %%
    descriptions_df['tokenid'] = descriptions_df['tokenid'].astype(float)

    # %%
    listings_with_names = listings_df.merge(descriptions_df, how='left', on='tokenid')
    listings_with_names

    # %%
    listings_with_names = listings_with_names.dropna()
    box_listings = listings_with_names[listings_with_names['name'].str.endswith('.box')]
    box_listings

    # %%
    events_df_copy = events_df.copy()
    events_df_copy

    # %%
    bids = events_df_copy[events_df_copy['event_type'] == 'order']
    sales = events_df_copy[events_df_copy['event_type'] == 'sale']

    bids

    # %%
    bids['identifier'] = bids['asset'].apply(lambda x: x.get('identifier') if isinstance(x, dict) else None)
    sales['identifier'] = sales['nft'].apply(lambda x: x.get('identifier') if isinstance(x, dict) else None)

    # %%
    bids['identifier'] = bids['identifier'].astype(float)
    bids.rename(columns={'identifier':'tokenid'}, inplace=True) 


    # %%
    bids = bids.merge(descriptions_df, how='left', on='tokenid')

    # %%
    bids.columns

    # %%
    bids.drop(columns=['protocol_address','chain','maker','criteria','is_private_listing','closing_date','nft','seller','buyer'], inplace=True)

    # %%
    sales['identifier'] = sales['identifier'].astype(float)
    sales.rename(columns={'identifier':'tokenid'}, inplace=True)


    # %%
    sales = sales.merge(descriptions_df, how='left', on='tokenid')
    sales.drop(columns=['order_type', 'chain','start_date','expiration_date','asset','maker','is_private_listing','protocol_address','criteria'], inplace=True)
    sales = sales.dropna()
    bids = bids.dropna()

    # %%
    box_sales_os = sales[sales['name'].str.endswith('.box')]
    box_bids_os = bids[bids['name'].str.endswith('.box')] 

    # %%
    box_bids_os.columns = [f'bid_{col}' if col != 'name' else col for col in box_bids_os.columns]

    # %%
    box_sales_os.columns = [f'sale_{col}' if col != 'name' else col for col in box_sales_os.columns]

    # %%
    box_listings_and_sales = pd.merge(box_bids_os, box_sales_os, how='inner', on='name')
    box_listings_and_sales['bid_event_timestamp'] = pd.to_datetime(box_listings_and_sales['bid_event_timestamp'], unit='s')
    box_listings_and_sales['sale_event_timestamp'] = pd.to_datetime(box_listings_and_sales['sale_event_timestamp'], unit='s')


    # %%
    filtered_box_listings_and_sales = box_listings_and_sales[box_listings_and_sales['sale_event_timestamp'] > box_listings_and_sales['bid_event_timestamp']]
    filtered_box_listings_and_sales['time_diff'] = filtered_box_listings_and_sales['sale_event_timestamp'] - filtered_box_listings_and_sales['bid_event_timestamp']


    # %%
    avg_time_to_sell = filtered_box_listings_and_sales['time_diff'].mean()
    print(avg_time_to_sell)

    # %%
    closest_listings = filtered_box_listings_and_sales.loc[filtered_box_listings_and_sales.groupby(['name', 'sale_event_timestamp'])['time_diff'].idxmin()]


    # %%
    closest_listings['sale_quantity'] = closest_listings['sale_payment'].apply(lambda x: int(x['quantity']) / 10**18)
    closest_listings['listing_quantity'] = closest_listings['bid_payment'].apply(lambda x: int(x['quantity']) / 10**18)


    # %%
    closest_listings[['bid_event_timestamp','listing_quantity']].head()

    # %%
    closest_listings['bid_event_timestamp'] = closest_listings['bid_event_timestamp'].dt.strftime('%Y-%m-%d %H:00:00')


    # %%
    closest_listings['sale_event_timestamp'] = closest_listings['sale_event_timestamp'].dt.strftime('%Y-%m-%d %H:00:00')

    # %%
    mint_df['tokenid'] = mint_df['tokenid'].astype(float)

    # %%
    mints_with_names = pd.merge(mint_df, descriptions_df, how='left', on='tokenid')

    # %%
    mints_with_names_null = mints_with_names[mints_with_names.isnull().any(axis=1)]
    print(list(mints_with_names_null['tx_hash']))

    # %%
    mints_with_names.drop_duplicates('tokenid', inplace=True)

    # %%
    mints_with_names.drop(columns=['__row_index','tx_hash','tokenid'], inplace=True)

    # %%
    mints_with_names.set_index('day', inplace=True)

    # %%
    mints_with_names.index = pd.to_datetime(mints_with_names.index)
    mints_with_names.dropna(inplace=True)

    # %%
    box_domains_mints = mints_with_names[mints_with_names['name'].str.endswith('.box')]


    # %%
    daily_box_mints = box_domains_mints.resample('D').count()

    # %%
    daily_box_mints.rename(columns={'name':'mints'}, inplace=True)
    daily_box_mints_fig = px.bar(daily_box_mints, x=daily_box_mints.index, y='mints', title='Daily Mints')
    # daily_box_mints_fig.show()

    # %%
    total_box_mints = box_domains_mints.count().iloc[0]
    total_box_mints

    # %%
    sales_with_names = pd.merge(sales_df, descriptions_df, how='left', on='tokenid')

    # %%
    sales_with_names.drop_duplicates('tokenid', inplace=True)
    sales_with_names.drop(columns=['__row_index','tx_hash','tokenid'], inplace=True)
    sales_with_names.set_index('day', inplace=True)
    sales_with_names.index = pd.to_datetime(sales_with_names.index)

    # %%
    sales_with_names

    # %% [markdown]
    # sales_with_names.dropna(inplace=True)

    # %%
    box_domains_sales = sales_with_names[sales_with_names['name'].str.endswith('.box')]
    box_domains_sales

    # %%
    eth_usd_df.set_index('day', inplace=True)
    eth_usd_df.index = pd.to_datetime(eth_usd_df.index)
    eth_usd_df.drop(columns=['__row_index'], inplace=True)

    # %%
    eth_usd_df.rename(columns={'price':'eth_usd'}, inplace=True)

    # %%
    box_listings['start_time'] = pd.to_datetime(box_listings['start_time'], unit='s').dt.strftime('%Y-%m-%d %H:00:00')
    box_listings['end_time'] = pd.to_datetime(box_listings['end_time'], unit='s').dt.strftime('%Y-%m-%d %H:00:00')

    # %%
    eth_usd_df_copy = eth_usd_df.reset_index().copy()
    eth_usd_df_copy.rename(columns={'day':'start_time'}, inplace=True)

    # %%
    eth_usd_df_copy['start_time'] = pd.to_datetime(eth_usd_df_copy['start_time']).dt.tz_localize(None)

    # %%
    box_listings['start_time'] = pd.to_datetime(box_listings['start_time']) 

    # %%
    eth_usd_df_copy_2 = eth_usd_df_copy.copy()
    eth_usd_df_copy_2.rename(columns={'start_time':'bid_event_timestamp', 'eth_usd':'eth_usd_bid'}, inplace=True)

    # %%
    eth_usd_df_copy_3 = eth_usd_df_copy.copy()
    eth_usd_df_copy_3.rename(columns={'start_time':'sale_event_timestamp', 'eth_usd':'eth_usd_sale'}, inplace=True)

    # %%
    closest_listings['bid_event_timestamp'] = pd.to_datetime(closest_listings['bid_event_timestamp'])
    closest_listings['sale_event_timestamp'] = pd.to_datetime(closest_listings['sale_event_timestamp'])

    # %%
    closest_listings = closest_listings.merge(eth_usd_df_copy_2, how='left', on='bid_event_timestamp')

    # %%
    closest_listings = closest_listings.merge(eth_usd_df_copy_3, how='left', on='sale_event_timestamp')

    # %%
    closest_listings['sale_usd'] = closest_listings['sale_quantity'] * closest_listings['eth_usd_sale']
    closest_listings['list_usd'] = closest_listings['listing_quantity'] * closest_listings['eth_usd_bid']

    # %%
    closest_listings['percent_change'] = (closest_listings['sale_usd'] - closest_listings['list_usd']) / closest_listings['list_usd'] * 100

    # %%
    listing_price_to_sale_avg_pct_change = closest_listings['percent_change'].mean()
    listing_price_to_sale_avg_pct_change = 0 if pd.isna(listing_price_to_sale_avg_pct_change) else listing_price_to_sale_avg_pct_change

    print(listing_price_to_sale_avg_pct_change)

    # %%

    box_listings = box_listings.merge(eth_usd_df_copy, how='left', on='start_time') 

    # %%
    box_listings.drop(columns=['order_hash'], inplace=True)

    # %%
    box_listings['price_in_usd_start_time'] = box_listings['price_in_eth'] * box_listings['eth_usd']

    # %%
    box_listings.set_index('start_time', inplace=True)
    box_listings_max_daily = box_listings['price_in_usd_start_time'].resample('D').max()

    # %%
    box_listings_num_daily = box_listings['name'].resample('D').count()
    total_box_listings = box_listings_num_daily.sum()
    total_box_listings

    # %%
    box_listings_min_daily = box_listings['price_in_usd_start_time'].resample('D').min()
    box_listings_avg_daily = box_listings['price_in_usd_start_time'].resample('D').mean()

    # %%
    box_listings_max_daily.fillna(0, inplace=True)
    box_listings_min_daily.fillna(0, inplace=True)
    box_listings_avg_daily.fillna(0, inplace=True)

    # %%
    box_listing_data = pd.merge(box_listings_num_daily.to_frame('listings'), box_listings_max_daily.to_frame('max_price'), left_index=True,
                                right_index=True, how='inner')

    # %%
    box_listing_data = box_listing_data.merge(box_listings_min_daily.to_frame('min_price'), left_index=True,
                                            right_index=True, how='inner')

    box_listing_data = box_listing_data.merge(box_listings_avg_daily.to_frame('avg_price'), left_index=True,
                                            right_index=True, how='inner')

    # %%
    monthly_listings = box_listings['name'].resample('M').count()
    monthly_listings

    # %%
    def monthly_listings_growth_rate(listings):
        previous_month = listings.shift(1)
        growth_rate = ((listings - previous_month) / previous_month) * 100
        return growth_rate

    # %%
    listings_growth_rate = monthly_listings_growth_rate(monthly_listings)


    # %%
    listings_growth_rate.dropna(inplace=True)
    listings_growth_rate

    # %%
    box_domains_sales = box_domains_sales.merge(eth_usd_df, left_index=True, right_index=True, how='left')

    # %%
    box_domains_sales['price_usd'] = box_domains_sales['price'] * box_domains_sales['eth_usd']
    box_domains_sales.rename(columns={'price':'price_eth'}, inplace=True)

    # %%
    box_domains_sales.drop(columns=['eth_usd'], inplace=True)


    # %%
    box_domains_sales.sort_index(inplace=True)
    box_domains_mints.sort_index(inplace=True)

    # %%
    box_domains_sales = box_domains_sales[['name', 'price_usd','price_eth']]


    # %%
    max_eth_sale = box_domains_sales['price_eth'].max()
    max_usd_sale = box_domains_sales['price_usd'].max()

    # Retrieve the corresponding timestamps
    max_eth_sale_row = box_domains_sales.loc[box_domains_sales['price_eth'].idxmax()]
    max_usd_sale_row = box_domains_sales.loc[box_domains_sales['price_usd'].idxmax()]

    # Display the results
    print(f"Maximum sale: \n {max_eth_sale_row}")


    # %%
    total_box_sales = box_domains_sales['name'].count()
    print(f'total .box domain sales as of {dt.datetime.today()} : {total_box_sales}')

    # %%
    daily_box_sales = box_domains_sales['name'].resample('D').count()
    daily_box_sales

    # %%
    daily_box_vol = box_domains_sales['price_usd'].resample('D').sum()
    cumulative_box_vol = daily_box_vol.cumsum()
    cumulative_box_vol

    # %%
    daily_box_sales_fig = px.bar(daily_box_sales.to_frame('sales'), x=daily_box_sales.index, y='sales', title='Daily Sales')
    # daily_box_sales_fig.show()

    # %%
    latest_box_domains_sales = box_domains_sales.iloc[-10:] 
    latest_box_domains_sales

    # %%
    latest_box_domains_mints = box_domains_mints.iloc[-10:]
    # latest_box_domains_mints

    # %%
    cumulative_box_mints = daily_box_mints.cumsum()
    cumulative_box_mints.rename(columns={'mints':'cumulative mints'}, inplace=True)

    # %%
    daily_mint_metrics = daily_box_mints.merge(cumulative_box_mints, left_index=True, right_index=True, how='inner')
    # daily_mint_metrics

    # %%
    daily_mint_metrics_fig = make_subplots(specs=[[{"secondary_y": True}]])

    daily_mint_metrics_fig.add_trace(
        go.Bar(
            x=daily_mint_metrics.index,
            y=daily_mint_metrics['mints'],
            name='Mints'
        ),
        secondary_y=False
    )

    daily_mint_metrics_fig.add_trace(
        go.Scatter(
            x=daily_mint_metrics.index,
            y=daily_mint_metrics['cumulative mints'],
            name='Cumulative Mints',
            mode='lines'
        ),
        secondary_y=True
    )

    daily_mint_metrics_fig.update_xaxes(title_text="Date")

    # daily_mint_metrics_fig.show()


    # %%
    listings_growth_rate

    # %%
    listings_growth_rate_fig = px.bar(listings_growth_rate.to_frame('Monthly Listings Growth Rate'), x=listings_growth_rate.index,
                                    y='Monthly Listings Growth Rate', title='Monthly Listings Growth Rate')

    # listings_growth_rate_fig.show()

    # %%
    cumulative_box_sales = daily_box_sales.cumsum()


    # %%
    monthly_max_sold = box_domains_sales['price_usd'].resample('M').max()
    monthly_min_sold = box_domains_sales['price_usd'].resample('M').min()
    monthly_avg_sold = box_domains_sales['price_usd'].resample('M').mean()
    monthly_volume_usd = box_domains_sales['price_usd'].resample('M').sum()
    monthly_num_sold = box_domains_sales['name'].resample('M').count()

    # %%
    monthly_max_sold.fillna(0, inplace=True)
    monthly_min_sold.fillna(0, inplace=True)
    monthly_avg_sold.fillna(0, inplace=True)
    monthly_volume_usd.fillna(0, inplace=True)
    monthly_num_sold.fillna(0, inplace=True)

    # %%
    monthly_box_sales_metrics = pd.merge(monthly_max_sold.to_frame('max_price'), monthly_min_sold.to_frame('min_price'), left_index=True, right_index=True, how='inner')

    # %%
    monthly_box_sales_metrics = monthly_box_sales_metrics.merge(monthly_avg_sold.to_frame('avg_price'), left_index=True, right_index=True, how='inner')

    # %%
    monthly_box_sales_metrics = monthly_box_sales_metrics.merge(monthly_volume_usd.to_frame('volume_usd'), left_index=True, right_index=True, how='inner')

    # %%
    monthly_num_sold

    # %%
    monthly_box_sales_metrics = monthly_box_sales_metrics.merge(monthly_num_sold.to_frame('domains sold'), left_index=True, right_index=True, how='inner')

    # %%
    monthly_box_sales_metrics

    # %%
    daily_sales_metrics = pd.merge(cumulative_box_sales.to_frame('cumulative_sales'), daily_box_sales.to_frame('daily_sales'), 
                                left_index=True, right_index=True, how='left')

    # %%
    daily_sales_metrics = daily_sales_metrics.merge(daily_box_vol.to_frame('vol_usd'), left_index=True, right_index=True, how='inner')
    daily_sales_metrics = daily_sales_metrics.merge(cumulative_box_vol.to_frame('cumulative_vol'), left_index=True, right_index=True, how='inner')

    # %%
    daily_sales_metrics

    # %%
    daily_vol_fig = make_subplots(specs=[[{"secondary_y": True}]])

    daily_vol_fig.add_trace(
        go.Bar(
            x=daily_sales_metrics.index,
            y=daily_sales_metrics['vol_usd'],
            name='Sales Volume'
        ),
        secondary_y=False
    )

    daily_vol_fig.add_trace(
        go.Scatter(
            x=daily_sales_metrics.index,
            y=daily_sales_metrics['cumulative_vol'],
            name='Cumulative Sales Volume',
            mode='lines'
        ),
        secondary_y=True
    )

    daily_vol_fig.update_xaxes(title_text="Date")

    # daily_vol_fig.show()


    # %%
    daily_sales_fig = make_subplots(specs=[[{"secondary_y": True}]])

    daily_sales_fig.add_trace(
        go.Bar(
            x=daily_sales_metrics.index,
            y=daily_sales_metrics['daily_sales'],
            name='Sales'
        ),
        secondary_y=False
    )

    daily_sales_fig.add_trace(
        go.Scatter(
            x=daily_sales_metrics.index,
            y=daily_sales_metrics['cumulative_sales'],
            name='Cumulative Sales',
            mode='lines'
        ),
        secondary_y=True
    )

    daily_sales_fig.update_xaxes(title_text="Date")

    # daily_sales_fig.show()


    # %%
    monthly_listings = monthly_listings.to_frame('listings')


    # %%
    monthly_sales = box_domains_sales['name'].resample('M').count()
    monthly_sales = monthly_sales.reset_index()
    monthly_sales['day'] = pd.to_datetime(monthly_sales['day']).dt.strftime('%Y-%m-%d')
    monthly_sales.set_index('day', inplace=True)

    # %%
    monthly_sales

    # %%
    monthly_listings.index

    # %%
    monthly_sales.index = pd.to_datetime(monthly_sales.index)

    # %%
    monthly_sales_reindexed = monthly_sales.reindex(monthly_listings.index).fillna(0)
    monthly_sales_reindexed


    # %%
    monthly_listings['sales'] = monthly_sales_reindexed['name']

    # %%
    monthly_listings

    # %%
    monthly_listings['listings_to_sales_ratio'] = monthly_listings['listings'] / monthly_listings['sales']
    monthly_listings['listings_to_sales_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)

    monthly_listings

    # %%
    listing_to_sales_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart for sales
    listing_to_sales_fig.add_trace(
        go.Bar(
            x=monthly_listings.index,
            y=monthly_listings['sales'],
            name='Sales'
        ),
        secondary_y=False
    )

    listing_to_sales_fig.add_trace(
        go.Bar(
            x=monthly_listings.index,
            y=monthly_listings['listings'],
            name='Listings'
        ),
        secondary_y=False
    )

    # Add line chart for cumulative sales
    listing_to_sales_fig.add_trace(
        go.Scatter(
            x=monthly_listings.index,
            y=monthly_listings['listings_to_sales_ratio'],
            name='Listings to Sales Ratio',
            mode='lines'
        ),
        secondary_y=True
    )

    # listing_to_sales_fig.show()


    # %%
    monthly_mints = box_domains_mints.resample('M').count()
    monthly_mints.reset_index(inplace=True)
    monthly_mints['day'] = pd.to_datetime(monthly_mints['day']).dt.strftime('%Y-%m-%d') 
    print(monthly_mints)

    # %%
    monthly_mints.set_index('day', inplace=True)


    # %%
    monthly_mints.index = pd.to_datetime(monthly_mints.index) 
    monthly_mints.index

    # %%
    monthly_sales_reindexed.index

    # %%
    monthly_mints['sales'] = monthly_sales_reindexed['name'] 
    monthly_mints.fillna(0, inplace=True)
    monthly_mints['mint_to_sales_ratio'] = monthly_mints['name'] / monthly_mints['sales'] 
    monthly_mints['mint_to_sales_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
    monthly_mints.rename(columns={'name':'mints'}, inplace=True)


    # %%
    monthly_mints

    # %%
    mint_to_sales_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart for sales
    mint_to_sales_fig.add_trace(
        go.Bar(
            x=monthly_mints.index,
            y=monthly_mints['sales'],
            name='Sales'
        ),
        secondary_y=False
    )

    mint_to_sales_fig.add_trace(
        go.Bar(
            x=monthly_mints.index,
            y=monthly_mints['mints'],
            name='Mints'
        ),
        secondary_y=False
    )

    # Add line chart for cumulative sales
    mint_to_sales_fig.add_trace(
        go.Scatter(
            x=monthly_mints.index,
            y=monthly_mints['mint_to_sales_ratio'],
            name='Mints to Sales Ratio',
            mode='lines'
        ),
        secondary_y=True
    )

    # mint_to_sales_fig.show()


    # %%
    cumulative_listings_to_sales = total_box_listings / total_box_sales
    print(cumulative_listings_to_sales)

    # %%
    cumulative_mint_to_sales = total_box_mints / total_box_sales
    print(cumulative_mint_to_sales)

    # %%
    import os
    print(os.getcwd())


    # %% [markdown]
    # ***Box Domains Valuation Model***

    # %% [markdown]
    # **Data Processing**

    # %% [markdown]
    # domain_path = 'E:/Projects/box_app/data/domain-name-sales.tsv'  
    # domain_data = pd.read_csv(domain_path, delimiter='\t')

    # %%
    domain_path = 'data/domain-name-sales.tsv'  
    domain_data = pd.read_csv(domain_path, delimiter='\t')


    # %%
    domain_data.set_index('date', inplace=True)
    domain_data = domain_data.drop(columns=['venue'])
    domain_data.sort_index(inplace=True)

    # %%
    domain_data.index = pd.to_datetime(domain_data.index)
    domain_data

    # %%
    domain_data['domain_length'] = domain_data['domain'].apply(len)
    domain_data['num_vowels'] = domain_data['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
    domain_data['num_consonants'] = domain_data['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
    domain_data['tld'] = domain_data['domain'].apply(lambda x: x.split('.')[-1])  # Extract TLD


    # %%
    domain_data

    # %%
    box_domains_sales.columns

    # %%
    filtered_box = box_domains_sales.drop(columns=['price_eth'])
    filtered_box.rename(columns={'name':'domain', 'price_usd':'price'}, inplace=True)


    # %%
    filtered_box['domain_length'] = filtered_box['domain'].apply(len)
    filtered_box['num_vowels'] = filtered_box['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
    filtered_box['num_consonants'] = filtered_box['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
    filtered_box['tld'] = filtered_box['domain'].apply(lambda x: x.split('.')[-1])  # Extract TLD


    # %%
    filtered_box.index = filtered_box.index.strftime('%Y-%m-%d')

    # %%
    filtered_box

    # %%
    features = ['domain_length', 'num_vowels', 'num_consonants', 'tld']
    X = domain_data[features]
    y = domain_data['price']

    # %%
    # Preprocess categorical data (TLD) and handle missing values
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ['domain_length', 'num_vowels', 'num_consonants']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['tld'])
        ]
    )

    # Create a pipeline with Ridge regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    }

    # %%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # %% [markdown]
    # **Ridge Regression**

    # %%
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Best Alpha: {grid_search.best_params_["regressor__alpha"]}')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')

    # %% [markdown]
    # **Random Forest Regressor**

    # %% [markdown]
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42))
    # ])
    # 
    # # Fit the model
    # pipeline.fit(X_train, y_train)
    # 
    # # Predict and evaluate
    # y_pred = pipeline.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # 
    # print(f'MAE: {mae}')
    # print(f'MSE: {mse}')
    # print(f'R²: {r2}')

    # %% [markdown]
    # **XGBoost**

    # %% [markdown]
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
    # ])
    # 
    # # Fit the model
    # pipeline.fit(X_train, y_train)
    # 
    # # Predict and evaluate
    # y_pred = pipeline.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # 
    # print(f'MAE: {mae}')
    # print(f'MSE: {mse}')
    # print(f'R²: {r2}')

    # %% [markdown]
    # **LightGBM**

    # %% [markdown]
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))
    # ])
    # 
    # # Fit the model
    # pipeline.fit(X_train, y_train)
    # 
    # # Predict and evaluate
    # y_pred = pipeline.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # 
    # print(f'MAE: {mae}')
    # print(f'MSE: {mse}')
    # print(f'R²: {r2}')

    # %% [markdown]
    # **Cat Boost**

    # %% [markdown]
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', CatBoostRegressor(iterations=200, depth=5, learning_rate=0.1, random_state=42, verbose=0))
    # ])
    # 
    # # Fit the model
    # pipeline.fit(X_train, y_train)
    # 
    # # Predict and evaluate
    # y_pred = pipeline.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # 
    # print(f'MAE: {mae}')
    # print(f'MSE: {mse}')
    # print(f'R²: {r2}')

    # %% [markdown]
    # **Prophet**

    # %% [markdown]
    # from sklearn.base import BaseEstimator, TransformerMixin
    # 
    # class ProphetRegressor(BaseEstimator, TransformerMixin):
    #     def __init__(self):
    #         self.model = Prophet()
    #         self.fitted_model = None
    # 
    #     def fit(self, X, y=None):
    #         df = pd.DataFrame({'ds': X.squeeze(), 'y': y})
    #         self.fitted_model = self.model.fit(df)
    #         return self
    # 
    #     def predict(self, X):
    #         future = pd.DataFrame({'ds': X.squeeze()})
    #         forecast = self.fitted_model.predict(future)
    #         return forecast['yhat'].values

    # %% [markdown]
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', ProphetRegressor())
    # ])

    # %% [markdown]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 
    # 

    # %% [markdown]
    # **Best Model**

    # %%
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1000.0))  # Set the best alpha value from grid search
    ])

    # %%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')

    # %%
    filtered_box

    # %%
    box_X = filtered_box[features]

    # Predict prices for .box domains using the best model
    filtered_box['predicted_price'] = pipeline.predict(box_X)

    print(filtered_box[['domain', 'predicted_price']])

    # %%
    r2 = r2_score(filtered_box['price'], filtered_box['predicted_price'])
    print(f'r2 {r2}')

    # %% [markdown]
    # **.Box Domain Valuator**

    # %%
    filtered_box_2 = filtered_box.drop(columns=['predicted_price'])
    filtered_box_2

    # %%
    combined_data = pd.concat([domain_data, filtered_box_2], ignore_index=True)

    # %%
    combined_data

    # %%
    X = combined_data[features]
    y = combined_data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')


    # %%
    def model_prep(data):
        data['domain_length'] = data['domain'].apply(len)
        data['num_vowels'] = data['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
        data['num_consonants'] = data['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
        data['tld'] = data['domain'].apply(lambda x: x.split('.')[-1]) 
        return data

    # %%
    def value_domain(domain):
        domain_x = domain[features]
        value = pipeline.predict(domain_x)
        print(f'predicted value: {value[0]}')
        return value[0] 

    # %%
    test_domain = 'eth.box' ## for model, just have the person input before .box, have it automatically add .box
    test_domain_df = pd.DataFrame({'domain': [test_domain]})
    test_domain_processed = model_prep(test_domain_df)
    test_domain_value = value_domain(test_domain_processed)

    # %%
    test_domain_value

    # %% [markdown]
    # ***Dash App***

    # %% [markdown]
    # Domain valuator would be callback

    # %%
    latest_box_domains_sales.reset_index(inplace=True)

    # %%
    latest_box_domains_sales.sort_values(by='day', ascending=False, inplace=True)

    # %%
    latest_box_domains_mints.reset_index(inplace=True)

    # %%
    latest_box_domains_mints.sort_values(by='day', ascending=False, inplace=True)

    # %%
    avg_box_sale = box_domains_sales['price_usd'].mean()

    # %%
    highest_selling_domains = box_domains_sales[['name','price_usd','price_eth']].sort_values(by='price_usd', ascending=False)
    highest_selling_domains = highest_selling_domains.head(10)
    highest_selling_domains

    # %%
    monthly_box_sales_metrics['cumulative_volume'] = monthly_box_sales_metrics['volume_usd'].cumsum()
    monthly_box_sales_metrics.reset_index(inplace=True) 
    monthly_box_sales_metrics.sort_values(by='day', ascending=False, inplace=True)

    # %%
    monthly_box_sales_metrics['cumulative domains sold'] = monthly_box_sales_metrics['domains sold'].cumsum()

    # %%
    historical_listing_to_sales = closest_listings[['name','bid_event_timestamp','sale_event_timestamp',
                                                    'list_usd','sale_usd','percent_change']]
    historical_listing_to_sales.sort_values(by='bid_event_timestamp', ascending=False, inplace=True)
    historical_listing_to_sales

    # %%
    box_listing_data.columns

    # %%
    box_listing_data.tail(20)

    # %%
    highest_selling_domains

    # %%
    highest_selling_domains_fig = px.bar(highest_selling_domains, x=highest_selling_domains['name'], y=highest_selling_domains['price_usd'],
                                        title='10 Highest Selling Domains')
    # highest_selling_domains_fig.show()

    # %%


    box_listing_data_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart for sales
    # box_listing_data_fig.add_trace(
    #     go.Bar(
    #         x=box_listing_data.index,
    #         y=box_listing_data['max_price'],
    #         name='Max Price'
    #     ),
    #     secondary_y=False
    # )

    # box_listing_data_fig.add_trace(
    #     go.Bar(
    #         x=box_listing_data.index,
    #         y=box_listing_data['min_price'],
    #         name='Min Price'
    #     ),
    #     secondary_y=False
    # )

    # Add line chart for cumulative sales
    box_listing_data_fig.add_trace(
        go.Bar(
            x=box_listing_data.index,
            y=box_listing_data['avg_price'],
            name='Avg Price',
        ),
        secondary_y=False
    )

    box_listing_data_fig.add_trace(
        go.Scatter(
            x=box_listing_data.index,
            y=box_listing_data['listings'],
            name='Listings',
            mode='lines'
        ),
        secondary_y=True
    )

    # box_listing_data_fig.show()


    # %%
    box_listings.drop(columns=['tokenid','eth_usd'], inplace=True)


    # %%
    latest_box_listings = box_listings.sort_index(ascending=False)
    latest_box_listings = latest_box_listings.head(10)
    latest_box_listings.reset_index(inplace=True)

    # %%
    mint_to_sales_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Mints to Sales Metrics"
    )

    listing_to_sales_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Listings to Sales Metrics"
    )

    daily_box_mints_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Daily Mints"
    )

    daily_box_sales_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Daily Sales"
    )

    daily_vol_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Daily Volume"
    )

    daily_sales_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Daily Sales Metrics"
    )

    daily_mint_metrics_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Daily Mints Metrics"
    )

    listings_growth_rate_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
        title="Listings Growth Rate"
    )

    highest_selling_domains_fig.update_layout(
        plot_bgcolor='#fafafa',
        paper_bgcolor='#fafafa',
    )



    # %% [markdown]
    # Average Time to Sell a .box Domain: 19 days 11:19:25.506097561
    # 
    # Average Listing Price to Sale Price Change -14.673687659116787
    # 
    # Cumulative Listings to Sales Ratio: 25:1
    # 
    # Cumulative Mints to Sales Ratio: 171:1
    # 
    # Monthly Listings Growth Rate: -70.2247191011236

    # %%
    avg_time_to_sell

    # %%
    key_metrics = [
        {"label": "Average Listing Price to Sale Price Change", "value": f"{round(int(listing_price_to_sale_avg_pct_change),0)}", "unit": "%"},
        {"label": "Average Days on Market", "value": str(avg_time_to_sell.days), "unit": " days"},
        {"label": "Cumulative Listings to Sales Ratio", "value": f"{round(int(cumulative_listings_to_sales), 0)}", "unit": ":1"},
        {"label": "Cumulative Mints to Sales Ratio", "value": f"{round(int(cumulative_mint_to_sales),0)}", "unit": ":1"},
        {"label": "Monthly Listings Growth Rate", "value": f"{listings_growth_rate.iloc[-1]:.2f}", "unit": "%"}
    ]


    # %%
    max_eth_sale_row

    # %%
    max_eth_sale_details = {
        "name": max_eth_sale_row["name"],
        "price_usd": max_eth_sale_row["price_usd"],
        "price_eth": max_eth_sale_row["price_eth"],
        "date": max_eth_sale_row.name  # This is the index (timestamp)
    }

    highest_sold_domain_str = f"""
    Name: {max_eth_sale_details['name']}
    Price (USD): ${max_eth_sale_details['price_usd']:.2f}
    Price (ETH): {max_eth_sale_details['price_eth']}
    Date: {max_eth_sale_details['date'].strftime('%Y-%m-%d')}
    """

    # %%
    highest_sold_domain_str

    # %%
    sales_metrics = [
        {"label": "Total Sales Volume ", "value":f"${cumulative_box_vol.iloc[-1]:,.2f}", "unit": ""},
        {"label": "Highest Sold Domain", "value": highest_sold_domain_str, "unit": ""},
        {"label": "Average Sales Price ", "value":f"${avg_box_sale:,.2f}", "unit": ""}
        
    ]

    # %%
    listings_metrics = [

        {"label":"Total .box Listings on Opensea ", "value": str(total_box_listings), "unit": ""}
    ]

    # %%
    mints_metrics = [
        {"label":"Total .box Mints ", "value":str(total_box_mints), "unit": ""}
    ]

    # %%
    def generate_table(dataframe, max_rows=11):
        return html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in dataframe.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ])
        ])

    return key_metrics, sales_metrics, listings_metrics, mints_metrics, mint_to_sales_fig, listing_to_sales_fig, daily_sales_fig, daily_vol_fig, highest_selling_domains_fig, monthly_box_sales_metrics, latest_box_domains_sales, highest_selling_domains, listings_growth_rate_fig, historical_listing_to_sales, latest_box_listings, daily_mint_metrics_fig, latest_box_domains_mints, model_prep, value_domain        


