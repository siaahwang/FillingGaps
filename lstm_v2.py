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
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置日志级别为 2（忽略 INFO 和 WARNING）
tf.get_logger().setLevel('ERROR')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

station_list = ['L3', 'L5', 'L7', 'L13', 'M3', 'M4', 'M7', 'M8']#'L3', 'L5', 'L7', 'L13', 'M3', 'M4', 'M7', 'M8', 'S4', 'S6'
metrics = []

for station in station_list:
    print(station)
    data = pd.read_csv(f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv")
    df = pd.DataFrame(data)
    data2 = pd.read_csv(f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_PP.csv")
    df2 = pd.DataFrame(data2)

    df['Soil_VWC_03'] = df['Soil_VWC_03'].replace(9999, np.nan)
    df['Prep'] = df2['Prep']
    df['value_filled'] = df['Soil_VWC_03'].fillna(0)
    df['mask'] = df['Soil_VWC_03'].isna().astype(int)
    scaler = StandardScaler()
    df['value_filled'] = scaler.fit_transform(df[['value_filled']])
    df['Prep'] = scaler.fit_transform(df[['Prep']])

    def create_sequences(df, seq_length):
        xs = []
        ys = []
        half_seq_length = seq_length // 2
        for i in range(half_seq_length, len(df) - half_seq_length):
            window = df.iloc[i - half_seq_length: i + half_seq_length + 1]
            if window['mask'].sum() == 0 and not np.isnan(df.iloc[i]['Soil_VWC_03']):  # 窗口内无缺失值且目标值不为缺失值
                x = window[['Prep', 'value_filled', 'mask']].drop(i).values  # 'Prep', 'value_filled', 'mask'
                xs.append(x)
                ys.append(df.iloc[i]['Soil_VWC_03'])
        return np.array(xs), np.array(ys)

    seq_length = 100
    X, y = create_sequences(df, seq_length)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(seq_length, 3)))  ################
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    X = X.reshape(X.shape[0], X.shape[1], 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

    #计算指标
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
    print("Train MAE:", mae_train)
    print("Test MAE:", mae_test)
    print("Train RMSE:", rmse_train)
    print("Test RMSE:", rmse_test)
    print("Train R2:", r2_train)
    print("Test R2:", r2_test)
    print("Train MRE:", mre_train)
    print("Test MRE:", mre_test)
    print("Train MAPE:", mape_train)
    print("Test MAPE:", mape_test)

    # 存储指标
    metrics.append([station, mae_train, rmse_train, r2_train, mre_train, mape_train, mae_test, rmse_test, r2_test, mre_test, mape_test])

    def impute_missing_values(df, model, seq_length):
        imputed = df.copy()
        half_seq_length = seq_length // 2
        for i in range(len(df)):
            if imputed['mask'].iloc[i] == 1:  # 只处理缺失值
                start_idx = max(0, i - half_seq_length)
                end_idx = min(len(df), i + half_seq_length + 1)
                window = imputed[['Prep', 'value_filled', 'mask']].iloc[start_idx:end_idx].drop(i).values  # 不包含目标值

                # 如果窗口长度不足，使用0值填充
                if len(window) < seq_length:
                    if start_idx == 0:
                        window = np.pad(window, ((seq_length - len(window), 0), (0, 0)), 'constant', constant_values=0)
                    else:
                        window = np.pad(window, ((0, seq_length - len(window)), (0, 0)), 'constant', constant_values=0)

                    #window[:, 0][window[:, 0] == 0] = np.nanmean(imputed['value_filled'])

                window = window.reshape((1, seq_length, 3))

                if not np.isnan(window).any():
                    prediction = model.predict(window, verbose=0)
                    prediction_value = prediction[0][0]

                    if not np.isnan(prediction_value):
                        imputed.loc[i, 'Soil_VWC_03'] = prediction_value
                        imputed.loc[i, 'mask'] = 0
                        imputed['value_filled'] = scaler.fit_transform(imputed[['Soil_VWC_03']].fillna(0))
        return imputed

    df_imputed = impute_missing_values(df, model, seq_length)
    df_imputed.to_csv(f'result/single/{station}_single_gapfilling.csv', index=True)

    # 绘制数据的图表
    """
    plt.figure(figsize=(35, 5))
    plt.rcParams.update({'font.size': 20})
    plt.title(f"{station}_2")
    plt.plot(df['Measure_Times'], df['Soil_VWC_03'], label='Actual Data', color='blue', linewidth=2.5)
    plt.plot(df_imputed['Measure_Times'], df_imputed['Soil_VWC_03'], label='Imputed Data', color='orange')
    plt.legend()
    plt.tight_layout()  # 自动调整子图参数以适应图形区域
    plt.subplots_adjust(left=0.02, right=0.97, top=0.95, bottom=0.1)  # 手动调整边距
    plt.show()
    plt.savefig(f"result/picture/{station}_plot.png")
    """
# 将指标写入Excel文件
metrics_df = pd.DataFrame(metrics, columns=['Station', 'Train_MAE', 'Train_RMSE', 'Train_R2', 'Train_MRE', 'Train_MAPE', 'Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MRE', 'Test_MAPE'])
metrics_df.to_excel(r'result/single/mae_rmse_r2_2.xlsx', index=False)