hour,
day,
weekday,
month,
weekend,
season,
ep_lag_1,
ep_lag_24,
ep_lag_168,
ep_rolling_mean_24,
dahtl_totalLoadValue,
dahtl_lag_1h,
dahtl_lag_24h,
dahtl_lag_168h,
dahtl_rolling_mean_24h,
dahtl_rolling_mean_168h,
atl_totalLoadValue,
atl_lag_1h,
atl_lag_24h,
atl_lag_168h,
atl_rolling_mean_24h,
atl_rolling_mean_168h,
temperature_2m,
wind_speed_10m,
wind_direction_10m,
cloudcover,
shortwave_radiation,
temperature_2m_lag1,
wind_speed_10m_lag1,
wind_direction_10m_lag1,
cloudcover_lag1,
shortwave_radiation_lag1,
temperature_2m_lag24,
wind_speed_10m_lag24,
wind_direction_10m_lag24,
cloudcover_lag24,
shortwave_radiation_lag24,
temperature_2m_lag168,
wind_speed_10m_lag168,
wind_direction_10m_lag168,
cloudcover_lag168,
shortwave_radiation_lag168,
ftc_DE_LU,
ftc_DK1,
ftc_GB,
ftc_NL,
Price[Currency/MWh]




'''
Old (42 features, eksl. date, zone, og price): 
python main_informer.py \
--model informer \
--data custom \
--root_path ./data/ \
--data_path filled_DK1_entso-e_combined_dataset_2019_2023.csv \
--features MS \
--target "Price[Currency/MWh]" \
--enc_in 42 --dec_in 42 --c_out 1 \
--seq_len 168 --label_len 24 --pred_len 24 \
--e_layers 2 --d_layers 1 --d_model 512 --n_heads 8 \
--train_epochs 10 --batch_size 32 --learning_rate 0.0001 \
--freq h --des "resources" \
--cols hour day weekday month weekend season ep_lag_1 ep_lag_24 ep_lag_168 ep_rolling_mean_24 \
dahtl_totalLoadValue dahtl_lag_1h dahtl_lag_24h dahtl_lag_168h dahtl_rolling_mean_24h dahtl_rolling_mean_168h \
atl_totalLoadValue atl_lag_1h atl_lag_24h atl_lag_168h atl_rolling_mean_24h atl_rolling_mean_168h \
temperature_2m wind_speed_10m wind_direction_10m cloudcover shortwave_radiation \
temperature_2m_lag1 wind_speed_10m_lag1 wind_direction_10m_lag1 cloudcover_lag1 shortwave_radiation_lag1 \
temperature_2m_lag24 wind_speed_10m_lag24 wind_direction_10m_lag24 cloudcover_lag24 shortwave_radiation_lag24 \
temperature_2m_lag168 wind_speed_10m_lag168 wind_direction_10m_lag168 cloudcover_lag168 shortwave_radiation_lag168


New (46 features, eksl. date, zone, og price): 
python main_informer.py \
--model informer \
--data custom \
--root_path ./data/ \
--data_path DK1_20250416.csv \
--features M \
--target "Price[Currency/MWh]" \
--enc_in 46 --dec_in 1 --c_out 1 \
--seq_len 168 --label_len 24 --pred_len 24 \
--e_layers 2 --d_layers 1 --d_model 512 --n_heads 8 \
--train_epochs 10 --batch_size 32 --learning_rate 0.0001 \
--freq h --des "resources" \
--cols hour day weekday month weekend season ep_lag_1 ep_lag_24 ep_lag_168 ep_rolling_mean_24 \ 10
dahtl_totalLoadValue dahtl_lag_1h dahtl_lag_24h dahtl_lag_168h dahtl_rolling_mean_24h dahtl_rolling_mean_168h \ 6
atl_totalLoadValue atl_lag_1h atl_lag_24h atl_lag_168h atl_rolling_mean_24h atl_rolling_mean_168h \ 6
temperature_2m wind_speed_10m wind_direction_10m cloudcover shortwave_radiation \ 5
temperature_2m_lag1 wind_speed_10m_lag1 wind_direction_10m_lag1 cloudcover_lag1 shortwave_radiation_lag1 \ 5
temperature_2m_lag24 wind_speed_10m_lag24 wind_direction_10m_lag24 cloudcover_lag24 shortwave_radiation_lag24 \ 5
temperature_2m_lag168 wind_speed_10m_lag168 wind_direction_10m_lag168 cloudcover_lag168 shortwave_radiation_lag168 \ 5
ftc_DE_LU ftc_DK1 ftc_GB ftc_NL 4


python run.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path ./data/ \
--data_path DK1_20250416.csv \
--model informer \
--features MS \
--target 'Price[Currency/MWh]' \
--seq_len 168 \
--label_len 48 \
--pred_len 24 \
--enc_in 48 \
--dec_in 48 \
--c_out 1 \
--des 'Exp_MS_price_forecast' \
--itr 1

python main_informer.py \
--model informer \
--data custom \
--root_path ./data/ \
--data_path DK1_20250416.csv \
--features MS \
--target "Price[Currency/MWh]" \
--enc_in 49 --dec_in 49 --c_out 1 \
--seq_len 168 --label_len 24 --pred_len 24 \
--e_layers 2 --d_layers 1 --d_model 512 --n_heads 8 \
--train_epochs 10 --batch_size 32 --learning_rate 0.0001 \
--freq h --des "using_all_49_no_cols"


--target "Price[Currency/MWh]" \ x
--enc_in 46 --dec_in 1 --c_out 1 \ x
--seq_len 168 --label_len 24 --pred_len 24 \x
--e_layers 2 --d_layers 1 --d_model 512 --n_heads 8 \x
--train_epochs 10 --batch_size 32 --learning_rate 0.0001 \ xx
--freq h --des "resources" 

python main_informer.py \
--model informer \
--data custom \
--root_path ./data/ \
--data_path DK1_20250416.csv \
--features MS \
--target "Price[Currency/MWh]" \
--enc_in 47 --dec_in 47 --c_out 1 \
--seq_len 168 --label_len 24 --pred_len 24 \
--e_layers 2 --d_layers 1 --d_model 512 --n_heads 8 \
--train_epochs 10 --batch_size 32 --learning_rate 0.0001 \
--freq h --des "using_all_47_excl_date_mapcode" \
--cols hour day weekday month weekend season \
ep_lag_1 ep_lag_24 ep_lag_168 ep_rolling_mean_24 \
dahtl_totalLoadValue dahtl_lag_1h dahtl_lag_24h dahtl_lag_168h dahtl_rolling_mean_24h dahtl_rolling_mean_168h \
atl_totalLoadValue atl_lag_1h atl_lag_24h atl_lag_168h atl_rolling_mean_24h atl_rolling_mean_168h \
temperature_2m wind_speed_10m wind_direction_10m cloudcover shortwave_radiation \
temperature_2m_lag1 wind_speed_10m_lag1 wind_direction_10m_lag1 cloudcover_lag1 shortwave_radiation_lag1 \
temperature_2m_lag24 wind_speed_10m_lag24 wind_direction_10m_lag24 cloudcover_lag24 shortwave_radiation_lag24 \
temperature_2m_lag168 wind_speed_10m_lag168 wind_direction_10m_lag168 cloudcover_lag168 shortwave_radiation_lag168 \
ftc_DE_LU ftc_DK1 ftc_GB ftc_NL Price[Currency/MWh]



05/05 klokken 2:00:
NOTE: command cols er med target og enc og dec er inkl. target. Command2 er brugt til at tælle både command cols og csv.
python main_informer.py \
--model informer \
--data custom \
--root_path ./data/ \
--data_path DK1-24-normalized.csv \
--features MS \
--target "Price[Currency/MWh]" \
--enc_in 63 --dec_in 63 --c_out 1 \
--seq_len 168 --label_len 24 --pred_len 24 \
--e_layers 2 --d_layers 1 --d_model 512 --n_heads 8 \
--train_epochs 10 --batch_size 32 --learning_rate 0.0001 \
--freq h --des "using_all_63" \
--cols hour day weekday month weekend season \
ep_lag_1 ep_lag_24 ep_lag_168 ep_rolling_mean_24 \
dahtl_totalLoadValue dahtl_lag_1h dahtl_lag_24h dahtl_lag_168h dahtl_rolling_mean_24h dahtl_rolling_mean_168h \
atl_totalLoadValue atl_lag_1h atl_lag_24h atl_lag_168h atl_rolling_mean_24h atl_rolling_mean_168h \
temperature_2m wind_speed_10m wind_direction_10m cloudcover shortwave_radiation \
temperature_2m_lag1 wind_speed_10m_lag1 wind_direction_10m_lag1 cloudcover_lag1 shortwave_radiation_lag1 \
temperature_2m_lag24 wind_speed_10m_lag24 wind_direction_10m_lag24 cloudcover_lag24 shortwave_radiation_lag24 \
temperature_2m_lag168 wind_speed_10m_lag168 wind_direction_10m_lag168 cloudcover_lag168 shortwave_radiation_lag168 \
ftc_DE_LU ftc_DK1 ftc_GB ftc_NL \
Natural_Gas_price_EUR Natural_Gas_price_EUR_lag_1d Natural_Gas_price_EUR_lag_7d Natural_Gas_rolling_mean_24h \
Coal_price_EUR Coal_price_EUR_lag_1d Coal_price_EUR_lag_7d Coal_rolling_mean_7d \
Oil_price_EUR Oil_price_EUR_lag_1d Oil_price_EUR_lag_7d Oil_rolling_mean_7d \
Carbon_Emission_price_EUR Carbon_Emission_price_EUR_lag_1d Carbon_Emission_price_EUR_lag_7d Carbon_Emission_rolling_mean_7d \
Price[Currency/MWh]


python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./resources \
  --data_path DK2_full_data_2018_2024.csv \
  --features MS \
  --target Electricity_price_MWh \
  --seq_len 720 \
  --label_len 168 \
  --pred_len 24 \
  --factor 3 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 --d_layers 1 \
  --enc_in 39 --dec_in 39 --c_out 1 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --dropout 0.1 \
  --train_epochs 40 --patience 8 \
  --num_workers 16 \
  --des "dk2"

  python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./resources \
  --data_path DK2_full_data_2018_2024.csv \
  --features MS \
  --target Electricity_price_MWh \
  --seq_len 720 \
  --label_len 168 \
  --pred_len 24 \
  --factor 3 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 1 --d_layers 1 \
  --enc_in 39 --dec_in 39 --c_out 1 \
  --batch_size 64 \
  --learning_rate 3e-4  \
  --dropout 0.1 \
  --train_epochs 40 --patience 8 \
  --num_workers 16 \
  --des "dk2_11"

python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./resources \
  --data_path DK2_full_data_2018_2024.csv \
  --features MS \
  --target Electricity_price_MWh \
  --seq_len 720  --label_len 168  --pred_len 24 \
  --factor 3 \
  --d_model 256  \
  --d_ff 512  \
  --n_heads 8 \
  --d_layers 1 \
  --enc_in 39   --dec_in 39  --c_out 1 \
  --batch_size 64  \
  --learning_rate 3e-4  \
  --dropout 0.1 \
  --num_workers 16 \
  --train_epochs 40  --patience 8 \
  --des "dk2_12"

  Næste som skal køres:
  python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./resources \
  --data_path DK2_full_data_2018_2024.csv \
  --features MS \
  --target Electricity_price_MWh \
  --seq_len 1440 \
  --label_len 336 \
  --pred_len 24 \
  --factor 3 \
  --d_model 512 \
  --d_ff 1024   \
  --n_heads 8 \
  --e_layers 4 --d_layers 1 \
  --enc_in 39 --dec_in 39 --c_out 1 \
  --batch_size 128 \
  --learning_rate 8e-4  \
  --dropout 0.2 \
  --train_epochs 40 --patience 8 \
  --num_workers 16 \
  --des "dk2_13"

  python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./resources \
  --data_path DK2_full_data_2018_2024.csv \
  --features MS \
  --target Electricity_price_MWh \
  --seq_len 1440 \
  --label_len 336 \
  --pred_len 24 \
  --factor 3 \
  --d_model 512 \
  --d_ff 1024   \
  --n_heads 8 \
  --e_layers 1 --d_layers 1 \
  --enc_in 39 --dec_in 39 --c_out 1 \
  --batch_size 64 \
  --learning_rate 3e-4  \
  --dropout 0.2 \
  --train_epochs 40 --patience 8 \
  --num_workers 16 \
  --des "dk2_14"

changed exp/expinformer

linje

  python -u main_informer.py \
  --model informer \
  --data custom \
  --root_path ./resources \
  --data_path DK2_full_data_2018_2024.csv \
  --features MS \
  --target Electricity_price_MWh \
  --seq_len 720  --label_len 168  --pred_len 24 \
  --factor 3 \
  --d_model 256  --d_ff 512  --n_heads 8 \
  --e_layers 3  --d_layers 1 \
  --enc_in 39   --dec_in 1   --c_out 1 \
  --batch_size 64  --learning_rate 3e-4 \
  --dropout 0.3 \
  --num_workers 16 \
  --train_epochs 40  --patience 8 \
  --des dk2_logtarget
'''