import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
#warnings.filterwarnings('ignore', category=UserWarning)
#warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

station_list = ['L3', 'L5', 'L7', 'L13', 'M3', 'M4', 'M7', 'M8', 'S4', 'S6']
for station in station_list:
    print(station)
    data = pd.read_csv(f"D:\\SM\\interpolation\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv")
    df = pd.DataFrame(data)
    # data2 = pd.read_csv(f"D:\\SM\\interpolation\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_PP.csv")
    # df2 = pd.DataFrame(data2)

    df['Soil_VWC_03'] = df['Soil_VWC_03'].replace(9999, np.nan)
    # df['Prep'] = df2['Prep']
    df['value_filled'] = df['Soil_VWC_03'].fillna(0)
    df['mask'] = df['Soil_VWC_03'].isna().astype(int)
    scaler = StandardScaler()
    df['value_filled'] = scaler.fit_transform(df[['value_filled']])
    # df['Prep'] = scaler.fit_transform(df[['Prep']])

    def create_sequences(df, seq_length):
        xs = []
        ys = []
        for i in range(len(df) - seq_length):
            x = df.iloc[i:(i + seq_length)][['value_filled', 'mask']].values  # 'Prep', 'value_filled', 'mask'
            y = df.iloc[i + seq_length]['Soil_VWC_03']
            if not np.isnan(y):  # 确保 y 不是 NaN
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)


    seq_length = 50  ####特征值长度
    X, y = create_sequences(df, seq_length)

    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(seq_length, 2)))  #################特征值种类
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

    # 计算训练集和测试集的 MAE 和 RMSE
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print("Test MAE:", mae_test)
    print("Test RMSE:", rmse_test)
    print("Train MAE:", mae_train)
    print("Train RMSE:", rmse_train)


    def impute_missing_values(df, model, seq_length):
        imputed = df.copy()
        for i in range(len(df)):
            if imputed['mask'].iloc[i] == 1:
                start = max(0, i - seq_length)
                end = i
                sequence = imputed.iloc[start:end][['value_filled', 'mask']].values  # 'Prep', 'value_filled', 'mask'
                if len(sequence) < seq_length:
                    sequence = np.pad(sequence, ((seq_length - len(sequence), 0), (0, 0)), mode='constant',
                                      constant_values=0)
                sequence = sequence.reshape((1, seq_length, 2))  ############################
                prediction = model.predict(sequence, verbose=0)
                prediction_value = prediction[0, 0]
                imputed.loc[imputed.index[i], 'Soil_VWC_03'] = prediction_value
                imputed.loc[imputed.index[i], 'mask'] = 0
                imputed['value_filled'] = scaler.fit_transform(imputed[['Soil_VWC_03']].fillna(0))
        return imputed


    df_imputed = impute_missing_values(df, model, seq_length)
    df_imputed.to_csv(f'result/{station}_imputed_data.csv', index=True)

    # 绘制数据的图表
    plt.figure(figsize=(35, 5))
    plt.rcParams.update({'font.size': 20})
    plt.title(f"{station}_singe")
    plt.plot(df['Measure_Times'], df['Soil_VWC_03'], label='Actual Data', color='blue', linewidth=2.5)
    plt.plot(df_imputed['Measure_Times'], df_imputed['Soil_VWC_03'], label='Imputed Data', color='orange')
    plt.legend()
    plt.tight_layout()  # 自动调整子图参数以适应图形区域
    plt.subplots_adjust(left=0.02, right=0.97, top=0.95, bottom=0.1)  # 手动调整边距
    plt.show()
