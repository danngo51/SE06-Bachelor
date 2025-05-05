# Import necessary libraries
import pandas as pd
import os

# Define folder paths
input_folder = "/Volumes/SSD/SEBachelor/entso-e data/DayAHeadTotalLoad/"  
output_folder = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/DayAHeadTotalLoad/"  
yearly_output_folder = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/DayAHeadTotalLoad/Yearly/"
full_output_file = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/DayAHeadTotalLoad/DayAHeadTotalLoad_FullDataset_2019_2023.csv"

# Initialize list to store processed DataFrames
processed_dfs = []
yearly_data = {}  # Dictionary to store data by year

# Ensure yearly output folder exists
os.makedirs(yearly_output_folder, exist_ok=True)

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".csv"):
        input_file_path = os.path.join(input_folder, filename)

        print(f"\n[INFO] Processing file: {filename}")

        # Load the CSV file (using tab delimiter)
        df = pd.read_csv(input_file_path, delimiter='\t')

        # Copy original dataframe to avoid modifying raw data
        df_clean = df.copy()

        # Step 1: Filter rows where "ResolutionCode" is not "PT60M"
        df_clean = df_clean[df_clean["ResolutionCode"] == "PT60M"]
        df_clean = df_clean[df_clean["AreaTypeCode"] == "BZN"]

        # Filter only selected countries
        CountriesToConclude = [
            "AT", "BE", "CH", "DE_LU","DE_50HzT", "DE_Amprion", "DE_TenneT_GER", "DE_TransnetBW", "DK1", "DK2", "ES", "FI", "FR", "GB", 
            "IE_SEM", "IT-CNORTH", "IT-CSOUTH", "IT-NORTH", "IT-Rossano", 
            "IT-SACOAC", "IT-SACODC", "IT-Sardinia", "IT-Sicily", "IT-SOUTH", 
            "NL", "NO1", "NO2", "NO3", "NO4", "NO5", "PT", "SE1", "SE2", "SE3", "SE4"
        ]
        df_clean = df_clean[df_clean["MapCode"].isin(CountriesToConclude)]

        # Step 2: Convert "DateTime(UTC)" to datetime and extract date
        df_clean["DateTime"] = pd.to_datetime(df_clean["DateTime"])  
        df_clean["Date"] = df_clean["DateTime"].dt.date  
        df_clean["Year"] = df_clean["DateTime"].dt.year  # Extract year

        # Step 3: Extract additional time-based features
        df_clean["hour"] = df_clean["DateTime"].dt.hour  
        df_clean["day"] = df_clean["DateTime"].dt.day
        df_clean["weekday"] = df_clean["DateTime"].dt.weekday
        df_clean["month"] = df_clean["DateTime"].dt.month
        df_clean["weekend"] = (df_clean["weekday"] >= 5).astype(int)  

        df_clean["season"] = df_clean["month"].map({12: 0, 1: 0, 2: 0,
                                                    3: 1, 4: 1, 5: 1,
                                                    6: 2, 7: 2, 8: 2,
                                                    9: 3, 10: 3, 11: 3})

        # **Step 4: Ensure chronological order within each country**
        df_clean = df_clean.sort_values(by=["MapCode", "DateTime", "AreaTypeCode"]).reset_index(drop=True)

        # **Step 5: Create lag features (grouped by MapCode)**
        df_clean["lag_1h"] = df_clean.groupby("MapCode")["TotalLoadValue"].shift(3) #1*3
        df_clean["lag_24h"] = df_clean.groupby("MapCode")["TotalLoadValue"].shift(72) #24*3
        df_clean["lag_168h"] = df_clean.groupby("MapCode")["TotalLoadValue"].shift(504) #168*3 

        # **Step 6: Create rolling mean features (grouped by MapCode)**
        df_clean["rolling_mean_24h"] = df_clean.groupby("MapCode")["TotalLoadValue"].rolling(window=24).mean().reset_index(level=0, drop=True)
        df_clean["rolling_mean_168h"] = df_clean.groupby("MapCode")["TotalLoadValue"].rolling(window=168).mean().reset_index(level=0, drop=True)

        # **Step 7: Fill missing values caused by shifting**
        df_clean.fillna(0, inplace=True)

        # Step 8: Remove unnecessary columns
        df_clean = df_clean.drop(columns=["DateTime",
                                        "ResolutionCode",
                                        "AreaCode",
                                        "AreaName",
                                        "UpdateTime"
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
    yearly_file_path = os.path.join(yearly_output_folder, f"DayAHeadTotalLoad_{year}.csv")
    yearly_df.to_csv(yearly_file_path, index=False)
    print(f"[INFO] Yearly dataset saved: {yearly_file_path}")

# **Step 9: Merge and Save the Full Dataset**
if processed_dfs:
    full_dataset = pd.concat(processed_dfs, ignore_index=True)
    full_dataset.to_csv(full_output_file, index=False)
    print(f"\n[INFO] Merged dataset saved successfully to: {full_output_file}")


print(f"\n[INFO] Preprocessed dataset saved at: {output_folder}")
