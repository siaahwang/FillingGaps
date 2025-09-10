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

station_list = ['L3', 'M3', 'M4', 'M7', 'M8', 'L5', 'L7', 'L13']
metrics = []
window_sizes = [50, 100, 150, 200, 250, 300]

station_additional_map = {
    'L3': ['S6', 'M2'],
    'M3': ['S2', 'S3', 'S4', 'S7', 'S8', 'M1', 'M4', 'M12'],
    'M4': ['M3', 'M5', 'M7', 'M11', 'M12', 'L7'],
    'M7': ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'M2', 'M4', 'M5', 'M6', 'M8', 'M9', 'M11'],
    'M8': ['M6', 'M7', 'M9', 'L4', 'L5', 'L13'],
    'L5': ['M8', 'L4'],
    'L7': ['M4', 'M5', 'M10', 'M11'],
    'L13': ['M8', 'M9', 'L11']
}

for station in station_list:
    print(station)
    data = pd.read_csv(f"30min_2019-2020/{station}_SM.csv")
    df = pd.DataFrame(data)
    data2 = pd.read_csv(f"30min_2019-2020/{station}_PP.csv")
    df2 = pd.DataFrame(data2)

    df['Soil_VWC_03'] = df['Soil_VWC_03'].replace(9999, np.nan)
    df['Prep'] = df2['Prep']
    df['value_filled'] = df['Soil_VWC_03'].fillna(0)
    df['mask'] = df['Soil_VWC_03'].isna().astype(int)
    scaler = StandardScaler()
    df['value_filled'] = scaler.fit_transform(df[['value_filled']])
    df['Prep'] = scaler.fit_transform(df[['Prep']])

    if station in station_additional_map:
        additional_stations = station_additional_map[station]
        for additional_station in additional_stations:
            additional_data = pd.read_csv(f"30min_2019-2020/{additional_station}_SM.csv")
            additional_df = pd.DataFrame(additional_data)
            additional_data_pp = pd.read_csv(f"30min_2019-2020/{additional_station}_PP.csv")
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
            if window['mask'].sum() == 0 and not np.isnan(df.iloc[i]['Soil_VWC_03']):
                if station in station_additional_map:
                    additional_features = []
                    for additional_station in station_additional_map[station]:
                        additional_features.extend([f'{additional_station}_Prep', f'{additional_station}_value_filled'])
                    x = window[['Prep', 'value_filled', 'mask'] + additional_features].drop(i).values
                else:
                    x = window[['Prep', 'value_filled', 'mask']].drop(i).values
                y = df.iloc[i]['Soil_VWC_03']
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)

    for seq_length in window_sizes:
        X, y = create_sequences(df, seq_length, station)

        model = Sequential()
        input_shape = (seq_length, 3) if station not in station_additional_map else (seq_length, 3 + 2 * len(station_additional_map[station]))
        model.add(Masking(mask_value=0, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        X = X.reshape(X.shape[0], X.shape[1], input_shape[1])
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
        print(f"Train MAE with window size {seq_length}:", mae_train)
        print(f"Test MAE with window size {seq_length}:", mae_test)
        print(f"Train RMSE with window size {seq_length}:", rmse_train)
        print(f"Test RMSE with window size {seq_length}:", rmse_test)
        print(f"Train R2 with window size {seq_length}:", r2_train)
        print(f"Test R2 with window size {seq_length}:", r2_test)
        print(f"Train MRE with window size {seq_length}:", mre_train)
        print(f"Test MRE with window size {seq_length}:", mre_test)
        print(f"Train MAPE with window size {seq_length}:", mape_train)
        print(f"Test MAPE with window size {seq_length}:", mape_test)

        metrics.append([station, seq_length, mae_train, rmse_train, r2_train, mre_train, mape_train, mae_test, rmse_test, r2_test, mre_test, mape_test])


        def impute_missing_values(df, model, seq_length, station):
            imputed = df.copy()
            half_seq_length = seq_length // 2
            for i in range(len(df)):
                if imputed['mask'].iloc[i] == 1:
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
                            imputed.loc[i, 'Soil_VWC_03'] = prediction_value
                            imputed.loc[i, 'mask'] = 0
                            imputed['value_filled'] = scaler.fit_transform(imputed[['Soil_VWC_03']].fillna(0))
            return imputed

        df_imputed = impute_missing_values(df, model, seq_length, station)
        df_imputed.to_csv(f'result/multi/{station}_multi_{seq_length}_gapfilling_0.85.csv', index=True)


        # plt.figure(figsize=(35, 5))
        # plt.rcParams.update({'font.size': 20})
        # plt.title(f"{station}_2")
        # plt.plot(df['Measure_Times'], df['Soil_VWC_03'], label='Actual Data', color='blue', linewidth=2.5)
        # plt.plot(df_imputed['Measure_Times'], df_imputed['Soil_VWC_03'], label='Imputed Data', color='orange')
        # plt.legend()
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.02, right=0.97, top=0.95, bottom=0.1)
        # plt.show()
        # plt.savefig(f"result/picture/{station}_plot.png")



    metrics_df = pd.DataFrame(metrics,
                              columns=['Station', 'Window_Size', 'Train_MAE', 'Train_RMSE',
                                       'Train_R2', 'Train_MRE', 'Train_MAPE', 'Test_MAE', 'Test_RMSE', 'Test_R2',
                                       'Test_MRE', 'Test_MAPE'])
    file_path = r'result/multi/mae_rmse_r2_0.85_v2.xlsx'
    if os.path.exists(file_path):
        existing_data = pd.read_excel(file_path)
        existing_data = pd.concat([existing_data, metrics_df], axis=0)
    else:
        existing_data = metrics_df
    existing_data.to_excel(file_path, index=False)