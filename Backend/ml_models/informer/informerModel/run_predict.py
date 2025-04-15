import torch
import pandas as pd
import numpy as np
from exp.exp_informer import Exp_Informer

# Match your original training args
class Args:
    model = 'informer'
    data = 'custom'
    root_path = './data/'
    data_path = 'test.csv' 
    checkpoints = './checkpoints/'
    features = 'MS'
    target = 'Price[Currency/MWh]'
    seq_len = 168
    label_len = 24
    pred_len = 24
    enc_in = 43
    dec_in = 43
    c_out = 1
    freq = 'h'
    embed = 'timeF'
    dropout = 0.2
    e_layers = 2
    d_layers = 1
    d_model = 512
    n_heads = 8
    d_ff = 2048
    attn = 'prob'
    factor = 5
    padding = 0
    distil = True
    activation = 'gelu'
    output_attention = False
    do_predict = True
    mix = True
    scale = False
    inverse = False
    flag = 'pred'
    do_predict = True
    cols = [  # same as your training cols
        'hour', 'day', 'weekday', 'month', 'weekend', 'season',
        'ep_lag_1', 'ep_lag_24', 'ep_lag_168', 'ep_rolling_mean_24',
        'dahtl_totalLoadValue', 'dahtl_lag_1h', 'dahtl_lag_24h', 'dahtl_lag_168h',
        'dahtl_rolling_mean_24h', 'dahtl_rolling_mean_168h',
        'atl_totalLoadValue', 'atl_lag_1h', 'atl_lag_24h', 'atl_lag_168h',
        'atl_rolling_mean_24h', 'atl_rolling_mean_168h',
        'temperature_2m', 'wind_speed_10m', 'wind_direction_10m',
        'cloudcover', 'shortwave_radiation',
        'temperature_2m_lag1', 'wind_speed_10m_lag1', 'wind_direction_10m_lag1',
        'cloudcover_lag1', 'shortwave_radiation_lag1',
        'temperature_2m_lag24', 'wind_speed_10m_lag24', 'wind_direction_10m_lag24',
        'cloudcover_lag24', 'shortwave_radiation_lag24',
        'temperature_2m_lag168', 'wind_speed_10m_lag168', 'wind_direction_10m_lag168',
        'cloudcover_lag168', 'shortwave_radiation_lag168', 'Price[Currency/MWh]'
    ]
    num_workers = 0
    use_gpu = torch.cuda.is_available()
    gpu = 0
    use_multi_gpu = False
    devices = '0'
    itr = 1
    des = 'inference'
    loss = 'mse'
    lradj = 'type1'
    use_amp = False

args = Args()

args.detail_freq = args.freq
args.freq = args.freq[-1:]


# Match folder name from training run
# Informer2020/results/informer_custom_ftMS_sl168_ll24_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_nomalized_test_run_1
setting = 'informer_custom_ftMS_sl168_ll24_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_nomalized_test_run_1'

exp = Exp_Informer(args)

# Run prediction (loads model checkpoint automatically)
print("Running prediction...")

preds, trues = exp.predict(setting, load=True)

# Save as CSV for easy plotting
np.savetxt(f'./results/{setting}/pred_prediction.csv', preds.reshape(preds.shape[0], -1), delimiter=',')
np.savetxt(f'./results/{setting}/real_prediction.csv', trues.reshape(trues.shape[0], -1), delimiter=',')



