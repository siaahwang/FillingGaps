import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
import os
warnings.filterwarnings('ignore')


station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']  # 'L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14'
metrics = []
metrics2 = []
depths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]#0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5


for station in station_list:
    print(f"Processing station: {station}")
    for depth in depths:
        depth_str = f"Soil_VWC_03_{depth}"
        print(f"Processing depth: {depth_str}")

        data = pd.read_csv(
            f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv")
        df = pd.DataFrame(data)

        # 预处理数据
        df['value_filled'] = df[depth_str].fillna(0)
        df['mask'] = df[depth_str].isna().astype(int)
        scaler = StandardScaler()
        df['value_filled'] = scaler.fit_transform(df[['value_filled']])

        gapfilled_df = pd.read_csv(f"result/test4/LSTM/{station}_gapfilling_{depth}.csv")

        # 计算缺失值所在位置的指标
        mask_missing = df['mask'] == 1
        y_true_missing = df.loc[mask_missing, 'Soil_VWC_03'].values
        y_pred_missing = gapfilled_df.loc[mask_missing, depth_str].values

        mae_gapfill_missing = mean_absolute_error(y_true_missing, y_pred_missing)
        rmse_gapfill_missing = np.sqrt(mean_squared_error(y_true_missing, y_pred_missing))
        r2_gapfill_missing = r2_score(y_true_missing, y_pred_missing)
        mape_gapfill_missing = np.mean(np.abs((y_true_missing - y_pred_missing) / y_true_missing)) * 100

        print(f"Station {station} Gapfilling Missing Value MAE:", mae_gapfill_missing)
        print(f"Station {station} Gapfilling Missing Value RMSE:", rmse_gapfill_missing)
        print(f"Station {station} Gapfilling Missing Value R2:", r2_gapfill_missing)
        print(f"Station {station} Gapfilling Missing Value MAPE:", mape_gapfill_missing)

        metrics2.append([station, depth, mae_gapfill_missing, rmse_gapfill_missing, r2_gapfill_missing, mape_gapfill_missing])
metrics2_df = pd.DataFrame(metrics2, columns=['Station', 'Rate', 'MAE', 'RMSE', 'R2', 'MAPE'])
file_path = r'result/test4/LSTM/mae_rmse_r2.xlsx'
if os.path.exists(file_path):
    existing_data = pd.read_excel(file_path)
    existing_data = pd.concat([existing_data, metrics2_df], axis=0)
else:
    existing_data = metrics2_df
existing_data.to_excel(file_path, index=False)



