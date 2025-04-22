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

'''