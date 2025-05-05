import pandas as pd

def denormalize_column(df, min_max_file, target_column):
    # Load min-max values
    min_max_df = pd.read_csv(min_max_file)

    if target_column not in min_max_df['Column'].values:
        raise ValueError(f"Target column '{target_column}' not found in min-max CSV!")

    # Get min and max for the target column
    row = min_max_df[min_max_df['Column'] == target_column].iloc[0]
    min_val = row['Min']
    max_val = row['Max']

    # De-normalization formula (inverse of MinMaxScaler)
    df_copy = df.copy()
    df_copy[target_column] = df_copy[target_column].apply(
        lambda x: x * (max_val - min_val) / 2 + (max_val + min_val) / 2 if -1 <= x <= 1 else x
    )

    return df_copy

if __name__ == "__main__":
    # Example usage

    # Paths
    normalized_dataset_path = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/entso-e_combined_dataset_2019_2023.csv"
    min_max_csv_path = "/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/normalization_minmax_values_with_buffer.csv"

    # Load your normalized dataset
    df_norm = pd.read_csv(normalized_dataset_path)

    # Specify which column you want to de-normalize
    target = "Price[Currency/MWh]"  # <-- change this!

    # De-normalize
    df_denorm = denormalize_column(df_norm, min_max_csv_path, target)

    # Save result
    df_denorm.to_csv("/Volumes/SSD/SEBachelor/entso-e data/Preproccessed/Results/denormalized_entso-e_combined_dataset_2019_2023.csv", index=False)
    print(f"[INFO] De-normalized dataset saved successfully.")
