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

def sliding_window(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[(i + window_size):(i + window_size + forecast_horizon)])
    return np.array(X), np.array(y)


window_sizes = [50]  # 可调整的窗口大小
forecast_horizon = 1

for station in station_list:
    print(f"Processing station: {station}")
    for depth in depths:
        depth_str = f"Soil_VWC_03_{depth}"
        print(f"Processing depth: {depth_str}")

        # 填补缺失值
        def impute_missing_values(df, model_fit, window_size):
            df_imputed = df.copy()
            for i in range(len(df)):
                if df_imputed.loc[i, 'mask'] == 1:  # 如果是缺失值
                    start_idx = max(0, i - window_size + 1)
                    end_idx = min(len(df), i + 1)
                    window = df_imputed[depth_str].iloc[start_idx:end_idx].values

                    # 如果窗口长度不足，使用0值填充
                    if len(window) < window_size:
                        window = np.pad(window, (window_size - len(window), 0), 'constant', constant_values=np.nan)

                    window = window.reshape((1, window_size))
                    prediction = model_fit.forecast(steps=1)[0]
                    df_imputed.loc[i, depth_str] = prediction
                    df_imputed.loc[i, 'mask'] = 0
                    df_imputed['value_filled'] = scaler.fit_transform(df_imputed[[depth_str]].fillna(0))
            return df_imputed

        data = pd.read_csv(f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test1\\{station}_SM.csv")
        df = pd.DataFrame(data)

        # 预处理数据
        df['value_filled'] = df[depth_str].fillna(0)
        df['mask'] = df[depth_str].isna().astype(int)
        scaler = StandardScaler()
        df['value_filled'] = scaler.fit_transform(df[['value_filled']])

        # 使用原始的Soil_VWC_03列进行训练
        training_data = df[depth_str].dropna().values

        # 参数搜索
        best_mae = np.inf
        best_order = None

        p_values = range(0, 6)
        d_values = [0, 1, 2]
        q_values = range(0, 6)

        for window_size in window_sizes:
            X, y = sliding_window(training_data, window_size, forecast_horizon)

            if X.shape[0] < 3:
                print(f"样本量不足，跳过深度 {depth} 的循环")
                continue
            print(f"X shape: {X.shape}, y shape: {y.shape}")

            # 划分训练集和测试集
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # 参数搜索
            best_mae = np.inf
            best_order = None

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = sm.tsa.ARIMA(X_train[:, -1], order=(p, d, q))
                            model_fit = model.fit()

                            # 对测试集进行预测
                            predictions = model_fit.forecast(steps=len(y_test))

                            # 计算测试集的 MAE
                            test_mae = mean_absolute_error(y_test, predictions)

                            if test_mae < best_mae:
                                best_mae = test_mae
                                best_order = (p, d, q)

                        except Exception as e:
                            print(f'Error for order ({p}, {d}, {q}): {e}')
                            continue

            if best_order is None:
                raise ValueError("No valid ARIMA model parameters found.")

            print(f'Window Size: {window_size} - Best Order: {best_order} - Best MAE: {best_mae}')

            # 使用最佳参数训练最终模型
            model = sm.tsa.ARIMA(X_train[:, -1], order=best_order)
            model_fit = model.fit()

            # 对测试集进行预测
            predictions = model_fit.forecast(steps=len(y_test))

            # 计算评估指标
            test_mae = mean_absolute_error(y_test, predictions)
            test_rmse = mean_squared_error(y_test, predictions, squared=False)
            test_r2 = r2_score(y_test, predictions)

            print(f"Window Size: {window_size}")
            print("Test MAE:", test_mae)
            print("Test RMSE:", test_rmse)
            print("Test R2:", test_r2)

            #metrics.append([station, window_size, best_order, test_mae, test_rmse, test_r2])

            df_imputed = impute_missing_values(df, model_fit, window_size)
            df_imputed.to_csv(f'result/test1/ARIMA/ARIMA_{station}_gapfilling_{depth}.csv', index=True)

            # 读取插补后的数据
            gapfilled_df = pd.read_csv(f"result/test1/ARIMA/ARIMA_{station}_gapfilling_{depth}.csv")

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
file_path = r'result/test1/ARIMA/ARIMA_mae_rmse_r2.xlsx'
if os.path.exists(file_path):
    existing_data = pd.read_excel(file_path)
    existing_data = pd.concat([existing_data, metrics_df], axis=0)
else:
    existing_data = metrics2_df
existing_data.to_excel(file_path, index=False)