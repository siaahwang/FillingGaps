import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
import os
import tensorflow as tf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']#'L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14'
metrics = []
metrics2 = []
window_sizes = [50]#50, 100, 150, 200, 250, 300
depths = [0.05]#0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5

station_additional_map = {
    'L4': ['M2', 'M8', 'L5'],
    'L6': ['L8'],
    'L8': ['M12', 'L6'],
    'L9': ['L10'],
    'L10': ['L9'],
    'L5': ['M8', 'L4'],
    'L11': ['M9', 'M10', 'L12', 'L13'],
    'L12': ['M10', 'L11']
}

for station in station_list:
    print(station)
    for depth in depths:
        depth_str = f"Soil_VWC_03_{depth}"
        print(f"Processing depth: {depth_str}")

        data = pd.read_csv(f"F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv")
        df = pd.DataFrame(data)
        data2 = pd.read_csv(f"F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_PP.csv")
        df2 = pd.DataFrame(data2)

        df['Prep'] = df2['Prep']
        df['value_filled'] = df[depth_str].fillna(0)
        df['mask'] = df[depth_str].isna().astype(int)
        scaler = StandardScaler()
        df['value_filled'] = scaler.fit_transform(df[['value_filled']])
        df['Prep'] = scaler.fit_transform(df[['Prep']])

        if station in station_additional_map:
            additional_stations = station_additional_map[station]
            for additional_station in additional_stations:
                additional_data = pd.read_csv(f"F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{additional_station}_SM.csv")
                additional_df = pd.DataFrame(additional_data)
                additional_data_pp = pd.read_csv(f"F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{additional_station}_PP.csv")
                additional_df_pp = pd.DataFrame(additional_data_pp)

                additional_df['Soil_VWC_03'] = additional_df['Soil_VWC_03'].replace(9999, np.nan)
                additional_df['Prep'] = additional_df_pp['Prep']
                additional_df['value_filled'] = additional_df['Soil_VWC_03'].fillna(0)
                additional_df['value_filled'] = scaler.fit_transform(additional_df[['value_filled']])
                additional_df['Prep'] = scaler.fit_transform(additional_df[['Prep']])

                df[f'{additional_station}_value_filled'] = additional_df['value_filled']
                df[f'{additional_station}_Prep'] = additional_df['Prep']

        def create_sequences(df, seq_length, station):
            xs = []
            ys = []
            half_seq_length = seq_length // 2
            for i in range(half_seq_length, len(df) - half_seq_length):
                window = df.iloc[i - half_seq_length: i + half_seq_length + 1]
                #if window['mask'].sum() == 0 and not np.isnan(df.iloc[i][depth_str]):
                if not np.isnan(df.iloc[i][depth_str]):
                    if station in station_additional_map:
                        additional_features = []
                        for additional_station in station_additional_map[station]:
                            additional_features.extend([f'{additional_station}_Prep', f'{additional_station}_value_filled'])
                        x = window[['Prep', 'value_filled', 'mask'] + additional_features].drop(i).values
                    else:
                        x = window[['Prep', 'value_filled', 'mask']].drop(i).values

                    if len(x) < seq_length:
                        if start_idx == 0:
                            x = np.pad(x, ((seq_length - len(window), 0), (0, 0)), 'constant', constant_values=0)
                        else:
                            x = np.pad(x, ((0, seq_length - len(window)), (0, 0)), 'constant', constant_values=0)
                    y = df.iloc[i][depth_str]
                    xs.append(x)
                    ys.append(y)
            return np.array(xs), np.array(ys)

        for seq_length in window_sizes:
            X, y = create_sequences(df, seq_length, station)

            if X.size == 0 or y.size == 0:
                print(f"Skipping {station} at depth {depth} due to insufficient data.")
                continue

            print(f"X shape: {X.shape}, y shape: {y.shape}")
            input_shape = (seq_length, X.shape[2])

            model = Sequential()#seq_length=50
            input_shape = (seq_length, 3) if station not in station_additional_map else (seq_length, 3 + 2 * len(station_additional_map[station]))
            model.add(Masking(mask_value=0, input_shape=input_shape))
            model.add(LSTM(50, return_sequences=True))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.summary()

            X = X.reshape(X.shape[0], X.shape[1], input_shape[1])
            if X.shape[0] < 5:
                print(f"样本量不足，使用所有数据进行训练: {station} at depth {depth}")
                X_train, y_train = X, y
                X_test, y_test = X, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            r2_train = r2_score(y_train, y_pred_train)
            mre_test = np.mean(np.abs((y_test - y_pred_test) / y_test))
            mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            mre_train = np.mean(np.abs((y_train - y_pred_train) / y_train))
            mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
            print(f"窗口大小 {seq_length} 的 Train MAE:", mae_train)
            print(f"窗口大小 {seq_length} 的 Test MAE:", mae_test)
            print(f"窗口大小 {seq_length} 的 Train RMSE:", rmse_train)
            print(f"窗口大小 {seq_length} 的 Test RMSE:", rmse_test)
            print(f"窗口大小 {seq_length} 的 Train R2:", r2_train)
            print(f"窗口大小 {seq_length} 的 Test R2:", r2_test)
            print(f"窗口大小 {seq_length} 的 Train MRE:", mre_train)
            print(f"窗口大小 {seq_length} 的 Test MRE:", mre_test)
            print(f"窗口大小 {seq_length} 的 Train MAPE:", mape_train)
            print(f"窗口大小 {seq_length} 的 Test MAPE:", mape_test)

            #metrics.append([station, depth, seq_length, mae_train, rmse_train, r2_train, mre_train, mape_train, mae_test, rmse_test, r2_test, mre_test, mape_test])


            def impute_missing_values(df, model, seq_length, station):
                imputed = df.copy()
                half_seq_length = seq_length // 2
                for i in range(len(df)):
                    if imputed['mask'].iloc[i] == 1:  # 只处理缺失值
                        start_idx = max(0, i - half_seq_length)
                        end_idx = min(len(df), i + half_seq_length + 1)
                        if station in station_additional_map:
                            additional_features = []
                            for additional_station in station_additional_map[station]:
                                additional_features.extend(
                                    [f'{additional_station}_Prep', f'{additional_station}_value_filled'])
                            window = imputed[['Prep', 'value_filled', 'mask'] + additional_features].iloc[
                                    start_idx:end_idx].drop(i).values
                        else:
                            window = imputed[['Prep', 'value_filled', 'mask']].iloc[start_idx:end_idx].drop(i).values

                        # 如果窗口长度不足，使用0值填充
                        if len(window) < seq_length:
                            if start_idx == 0:
                                window = np.pad(window, ((seq_length - len(window), 0), (0, 0)), 'constant',
                                            constant_values=0)
                            else:
                                window = np.pad(window, ((0, seq_length - len(window)), (0, 0)), 'constant',
                                            constant_values=0)

                        window = window.reshape((1, seq_length, window.shape[1]))

                        if not np.isnan(window).any():
                            prediction = model.predict(window, verbose=0)
                            prediction_value = prediction[0][0]

                            if not np.isnan(prediction_value):
                                imputed.loc[i, depth_str] = prediction_value
                                imputed.loc[i, 'mask'] = 0
                                imputed['value_filled'] = scaler.fit_transform(imputed[[depth_str]].fillna(0))
                return imputed

            df_imputed = impute_missing_values(df, model, seq_length, station)
            df_imputed.to_csv(f'result/test4/LSTM/{station}_gapfilling_{depth}.csv', index=True)

            # 读取插补后的数据
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
