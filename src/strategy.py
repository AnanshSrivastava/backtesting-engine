def generate_signals(df,SMA_short=20, SMA_long=50, verbose=True):
    """
    Generate trading signals based on the defined strategy.
    
    Returns:
        signals (list): A list of generated trading signals.
    """
    df['SMA_short'] = df['Adj Close'].rolling(window=SMA_short).mean()
    df['SMA_long'] = df['Adj Close'].rolling(window=SMA_long).mean()
    df['short_over_long'] = df['SMA_short'] > df['SMA_long']
    df['signal'] = df['short_over_long'].astype(int).diff()  # Convert boolean to integer (1 for True, 0 for False)

    crossovers = df[df['signal'].isin([1, -1])]
    if verbose:
        print(crossovers[['Adj Close', 'SMA_short', 'SMA_long', 'signal']])  # Print the crossover points for verification
    #actual signal generation
        for index, row in crossovers.iterrows():
            if row['signal'] == 1:
                print("Buy signal generated.")
            elif row['signal'] == -1:
                print("Sell signal generated.")
        print(df['signal'].tail(20))  # Print the last few rows of the DataFrame to verify the calculations
    
    return df
