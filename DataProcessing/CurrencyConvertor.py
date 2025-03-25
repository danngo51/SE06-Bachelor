import pandas as pd

def process_exchange_rates(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Rename columns for easier reference
    df.columns = ["DATE", "TIME_PERIOD", "EUR_to_GBP"]
    
    # Convert DATE column to datetime format
    df["DATE"] = pd.to_datetime(df["DATE"])
    
    # Filter dates between 2019-01-01 and 2023-12-31
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    df_filtered = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]
    
    # Reverse the exchange rate (GBP to EUR)
    df_filtered["GBP_to_EUR"] = 1 / df_filtered["EUR_to_GBP"]
    
    # Create a full date range
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_full = pd.DataFrame(full_date_range, columns=["DATE"])
    
    # Merge with filtered data to include all dates
    df_final = df_full.merge(df_filtered, on="DATE", how="left")
    
    # Forward-fill missing values with the last available exchange rate
    df_final["GBP_to_EUR"] = df_final["GBP_to_EUR"].ffill()
    
    # If the first row (2019-01-01) is still NaN, fill it with the first available value
    if pd.isna(df_final.loc[0, "GBP_to_EUR"]):
        first_valid_index = df_final["GBP_to_EUR"].first_valid_index()
        df_final.loc[0, "GBP_to_EUR"] = df_final.loc[first_valid_index, "GBP_to_EUR"]
    
    # Select relevant columns
    df_final = df_final[["DATE", "GBP_to_EUR"]]
    
    # Save the new dataset to a CSV file
    df_final.to_csv(output_file, index=False)
    print(f"Processed file saved as: {output_file}")

# Example usage
input_file = "/Volumes/SSD/SEBachelor/entso-e data/HistoricalExchangeRates/ECB Data Portal_20250318120736.csv"
output_file = "/Volumes/SSD/SEBachelor/entso-e data/HistoricalExchangeRates/GBP_to_EUR/GBP_to_EUR_rates_2019_2023.csv"
process_exchange_rates(input_file, output_file)