import pandas as pd

df = pd.read_csv('ml_models/data/DK1/DK1_full_data_2025.csv', parse_dates=['date'])
target_data = df[df['date'] == '2025-03-01']

print(f'Data available for 2025-03-01: {len(target_data)} rows')
if len(target_data) > 0:
    print('Hours available:', sorted(target_data['hour'].values))
    print('Price values:', target_data['Electricity_price_MWh'].values[:5])
else:
    print('No data found for 2025-03-01')

print(f'Last date in dataset: {df["date"].max()}')
print('Last few entries:')
print(df[['date', 'hour', 'Electricity_price_MWh']].tail())
