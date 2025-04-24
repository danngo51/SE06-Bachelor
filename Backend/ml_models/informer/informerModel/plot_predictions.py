import pandas as pd
import matplotlib.pyplot as plt

# Load files
real = pd.read_csv('./results/informer_custom_ftMS_sl168_ll24_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_year_split_exp_1/real_prediction.csv', header=None)
pred = pd.read_csv('./results/informer_custom_ftMS_sl168_ll24_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_year_split_exp_1/pred_prediction.csv', header=None)

# Plot a few examples (e.g., first batch)
for i in range(min(5, len(real))):
    plt.figure(figsize=(12, 4))
    plt.plot(real.iloc[i], label='Actual')
    plt.plot(pred.iloc[i], label='Predicted')
    plt.title(f'Sample {i+1}')
    plt.legend()
    plt.grid(True)
    plt.show()
