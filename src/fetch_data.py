import yfinance as yf

"""fetching data from yahoo finance and saving it to csv file while also checking for missing values in the data"""
def fetch_data(ticker, start_date, end_date):
    '''checks for error during data fetching'''
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
        print(f"Error occurred while fetching data for {ticker}: {e}")
        return None

    if df.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return None

    df.columns = df.columns.get_level_values(0)
    df.to_csv(f"data/{ticker}_{start_date}_{end_date}.csv")
    return df


"""fetching data for multiple tickers using fetch_data function"""
def fetch_multiple(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        df = fetch_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df
    return data


