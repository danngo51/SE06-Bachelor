# Import necessary libraries
import pandas as pd
import os

# Define folder paths
input_folder = "/Volumes/SSD/SEBachelor/entso-e data/EnergyPrices/"  
output_folder = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/EnergyPrices/"  
yearly_output_folder = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/EnergyPrices/Yearly/"
full_output_file = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/EnergyPrices/EnergyPrices_FullDataset_2019_2023.csv"

exchange_rate_file = "/Volumes/SSD/SEBachelor/entso-e data/HistoricalExchangeRates/GBP_to_EUR/GBP_to_EUR_rates_2019_2023.csv"

# Load historical GBP to EUR exchange rates
df_exchange_rates = pd.read_csv(exchange_rate_file)
df_exchange_rates["DATE"] = pd.to_datetime(df_exchange_rates["DATE"])  # Ensure proper datetime format

# Initialize list to store processed DataFrames
processed_dfs = []
yearly_data = {}  # Dictionary to store data by year

# Ensure yearly output folder exists
os.makedirs(yearly_output_folder, exist_ok=True)

# **Process files in sorted order**
for filename in sorted(os.listdir(input_folder)):  
    if filename.endswith(".csv"):  
        input_file_path = os.path.join(input_folder, filename)

        print(f"\n[INFO] Processing file: {filename}")

        # Load the CSV file (using tab delimiter)
        df = pd.read_csv(input_file_path, delimiter='\t')

        # Copy original dataframe to avoid modifying raw data
        df_clean = df.copy()

        # Step 1: Filter rows where "ResolutionCode" is "PT60M"
        df_clean = df_clean[df_clean["ResolutionCode"] == "PT60M"]

        # Filter only selected countries
        CountriesToConclude = [
            "AT", "BE", "CH", "DE_LU", "DK1", "DK2", "ES", "FI", "FR", "GB", 
            "IE_SEM", "IT-CNORTH", "IT-CSOUTH", "IT-NORTH", "IT-Rossano", 
            "IT-SACOAC", "IT-SACODC", "IT-Sardinia", "IT-Sicily", "IT-SOUTH", 
            "NL", "NO1", "NO2", "NO3", "NO4", "NO5", "PT", "SE1", "SE2", "SE3", "SE4"
        ]
        df_clean = df_clean[df_clean["MapCode"].isin(CountriesToConclude)]

        # Step 2: Convert "DateTime(UTC)" to "Date" (keeping only the date)
        df_clean["DateTime(UTC)"] = pd.to_datetime(df_clean["DateTime(UTC)"])  
        df_clean["Date"] = df_clean["DateTime(UTC)"].dt.date  
        df_clean["Year"] = df_clean["DateTime(UTC)"].dt.year  # Extract year

        # Step 3: Extract additional time-based features
        df_clean["hour"] = df_clean["DateTime(UTC)"].dt.hour  
        df_clean["day"] = df_clean["DateTime(UTC)"].dt.day
        df_clean["weekday"] = df_clean["DateTime(UTC)"].dt.weekday
        df_clean["month"] = df_clean["DateTime(UTC)"].dt.month
        df_clean["weekend"] = (df_clean["weekday"] >= 5).astype(int)  

        df_clean["season"] = df_clean["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                            3: "Spring", 4: "Spring", 5: "Spring",
                                            6: "Summer", 7: "Summer", 8: "Summer",
                                            9: "Fall", 10: "Fall", 11: "Fall"})

        # Step 4: Convert "GBP" Prices to "EUR" Using Historical Exchange Rates
        df_clean["Date"] = pd.to_datetime(df_clean["Date"])  
        df_clean = df_clean.merge(df_exchange_rates, left_on="Date", right_on="DATE", how="left")  

        # Convert GBP prices to EUR
        df_clean.loc[df_clean["Currency"] == "GBP", "Price[Currency/MWh]"] *= df_clean["GBP_to_EUR"]
        df_clean["Currency"] = "EUR"

        # Drop temporary exchange rate columns
        df_clean = df_clean.drop(columns=["GBP_to_EUR", "DATE"])

        # Step 5: Create Lag Features for Time-Series Forecasting
        df_clean["lag_1"] = df_clean["Price[Currency/MWh]"].shift(1).fillna(0)
        df_clean["lag_24"] = df_clean["Price[Currency/MWh]"].shift(24).fillna(0)
        df_clean["rolling_mean_24"] = df_clean["Price[Currency/MWh]"].rolling(window=24).mean().fillna(0)

        # Remove unnecessary columns
        df_clean = df_clean.drop(columns=["DateTime(UTC)",
                                          "AreaDisplayName",
                                          "AreaTypeCode",
                                          "Sequence", 
                                          "UpdateTime(UTC)", 
                                          "ContractType",
                                          "ResolutionCode",
                                          "AreaCode"
        ]) 

        # Append processed DataFrame to list
        processed_dfs.append(df_clean)

        # Step 7: Store yearly data separately
        for year, df_year in df_clean.groupby("Year"):
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].append(df_year)

# **Step 8: Merge and Save the Yearly Files**
for year, dfs in yearly_data.items():
    yearly_df = pd.concat(dfs, ignore_index=True)
    yearly_file_path = os.path.join(yearly_output_folder, f"EnergyPrices_{year}.csv")
    yearly_df.to_csv(yearly_file_path, index=False)
    print(f"[INFO] Yearly dataset saved: {yearly_file_path}")

# **Step 9: Merge and Save the Full Dataset**
if processed_dfs:
    full_dataset = pd.concat(processed_dfs, ignore_index=True)
    full_dataset.to_csv(full_output_file, index=False)
    print(f"\n[INFO] Merged dataset saved successfully to: {full_output_file}")

print("\nAll files have been successfully preprocessed, merged, and saved!")
