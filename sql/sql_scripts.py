def daily_mints ():
    data = """"
    with mints as(

    SELECT *
    FROM optimism.nft.ez_nft_transfers
    where nft_address = lower('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17') AND
    event_type = 'mint' AND ERC1155_VALUE IS NOT NULL
    order by block_timestamp desc

    )

    select date_trunc('day', block_timestamp) day, count(distinct tokenid) as number_of_mints
    from mints
    group by date_trunc('day', block_timestamp)
    order by date_trunc('day', block_timestamp) desc
    

    """
    return data

def mint_to_sales ():
    data = """
WITH mints AS (
  SELECT
    block_timestamp,
    tokenid
  FROM
    optimism.nft.ez_nft_transfers
  WHERE
    nft_address = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'mint'
    AND ERC1155_VALUE IS NOT NULL
),
desc_mint as (
  select
    d.*
  from
    mints d
),
daily_mints as (
  select
    date_trunc('day', block_timestamp) as day,
    count(distinct tokenid) as mints
  from
    desc_mint
  group by
    date_trunc('day', block_timestamp)
  order by
    day desc
),
sales_per_day AS (
  SELECT
    date_trunc('day', BLOCK_TIMESTAMP) day,
    count(tx_hash) total_sales,
  FROM
    optimism.nft.ez_nft_sales d
  WHERE
    d.NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND d.event_type = 'sale'
  group by
    day
  order by
    day desc
),
cumulative_counts AS (
  SELECT
    (
      SELECT
        SUM(mints)
      FROM
        daily_mints
    ) AS total_mints,
    (
      SELECT
        SUM(total_sales)
      FROM
        sales_per_day
    ) AS total_sales
),
cumulative_mints_to_sales_ratio AS (
  SELECT
    total_mints,
    total_sales,
    CASE
      WHEN total_sales > 0 THEN total_mints :: FLOAT / total_sales
      ELSE NULL
    END AS ratio
  FROM
    cumulative_counts
)
SELECT
  *
FROM
  cumulative_mints_to_sales_ratio;
    
    
    
    """
    return data

def sales_aggregate_metrics():
    data = """
    WITH sales AS (
  SELECT
    *,
    DATE_TRUNC('DAY', BLOCK_TIMESTAMP) AS day
  FROM
    optimism.nft.ez_nft_sales
  WHERE
    NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'sale'
),
weth_price as (
  select
    date_trunc('day', hour) as day,
    avg(price) as avg_price
  from
    ethereum.price.ez_prices_hourly
  where
    symbol = 'WETH'
  group by
    date_trunc('day', hour)
)
SELECT
  MAX(S.PRICE * W.AVG_PRICE) AS max_sold_usd,
  MAX(S.PRICE) AS MAX_SOLD_ETH,
  MIN(s.PRICE) AS min_sold_eth,
  MIN(S.PRICE * W.AVG_PRICE) AS MIN_SOLD_USD,
  AVG(s.PRICE) AS avg_price_eth,
  AVG(s.price * w.avg_price) as avg_price_usd,
  COUNT(EZ_NFT_SALES_ID) AS num_sold,
  SUM(price) AS volume_eth,
  sum(s.price * w.avg_price) AS VOLUME_USD
FROM
  sales s
  JOIN WETH_PRICE W ON S.DAY = W.DAY
    
    
    """
    return data

def daily_sales_metrics():
    data = """
  WITH sales AS (
    SELECT
      *,
      DATE_TRUNC('DAY', BLOCK_TIMESTAMP) AS day
    FROM
      optimism.nft.ez_nft_sales
    WHERE
      NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
      AND event_type = 'sale'
  ),
  weth_price AS (
    SELECT
      date_trunc('day', hour) AS day,
      AVG(price) AS avg_price
    FROM
      ethereum.price.ez_prices_hourly
    WHERE
      symbol = 'WETH'
    GROUP BY
      date_trunc('day', hour)
  ),
  combined as (
    select
      s.day,
      s.tokenid,
      s.price,
      (s.price * p.avg_price) as price_usd,
      s.tx_hash
    from
      sales s
      join weth_price p on p.day = s.day
  ),
  daily_stats AS (
    SELECT
      day,
      MAX(PRICE_USD) AS max_sold_usd,
      MIN(PRICE_USD) AS min_sold_usd,
      AVG(PRICE_USD) AS avg_price_usd,
      COUNT(tx_hash) AS num_sold,
      SUM(PRICE_USD) AS volume_usd
    FROM
      combined
    GROUP BY
      day
  ),
  cumulative_stats AS (
    SELECT
      day,
      max_sold_usd,
      min_sold_usd,
      avg_price_usd,
      num_sold,
      volume_usd,
      SUM(num_sold) OVER (
        ORDER BY
          day
      ) AS cumulative_num_sold,
      SUM(volume_usd) OVER (
        ORDER BY
          day
      ) AS cumulative_volume_usd
    FROM
      daily_stats
  )
  SELECT
    *
  FROM
    cumulative_stats
  ORDER BY
    day DESC;


  """
    return data

def min_max_stats():
    data="""
  WITH sales AS (
  SELECT
    *,
    DATE_TRUNC('DAY', BLOCK_TIMESTAMP) AS day
  FROM
    optimism.nft.ez_nft_sales
  WHERE
    NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'sale'
),
weth_price as (
  select
    date_trunc('day', hour) as day,
    avg(price) as avg_price
  from
    ethereum.price.ez_prices_hourly
  where
    symbol = 'WETH'
  group by
    date_trunc('day', hour)
)
SELECT
  MAX(S.PRICE * W.AVG_PRICE) AS max_sold_usd,
  MAX(S.PRICE) AS MAX_SOLD_ETH,
  MIN(s.PRICE) AS min_sold_eth,
  MIN(S.PRICE * W.AVG_PRICE) AS MIN_SOLD_USD,
  AVG(s.PRICE) AS avg_price_eth,
  AVG(s.price * w.avg_price) as avg_price_usd,
  COUNT(EZ_NFT_SALES_ID) AS num_sold,
  SUM(price) AS volume_eth,
  sum(s.price * w.avg_price) AS VOLUME_USD
FROM
  sales s
  JOIN WETH_PRICE W ON S.DAY = W.DAY



"""
    return data

