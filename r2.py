import pandas as pd
import numpy as np

#IDW
r2 = []
station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
depths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for station in station_list:
    for depth in depths:
        depth_str = f'Soil_VWC_03_{depth}'

        station_df = pd.read_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv',
                                        parse_dates=['Measure_Times'], index_col='Measure_Times', na_values=np.nan)
        df2 = pd.read_csv(f'result/test4/IDW/{station}_SM_interpolated_{depth}.csv',
                          parse_dates=['Measure_Times'], index_col='Measure_Times')

        # 找到当前深度的缺失值索引
        missing_indices = station_df[depth_str].isna()
        missing_times = station_df.index[missing_indices]

        result = []

        for time in missing_times:
            true_value = station_df.loc[time, 'Soil_VWC_03']
            interpolated_value = df2.loc[time, depth_str]
            if not np.isnan(true_value):
                result.append({'Time': time, 'True': true_value, 'Predict': interpolated_value})

        result_df = pd.DataFrame(result)
        correlation = result_df['True'].corr(result_df['Predict'])
        r2.append([station, depth, correlation])
        result_df.to_csv(f'result/test4/IDW2/{station}_r2_{depth}.csv', index=False)
r2_df = pd.DataFrame(r2, columns=['Station', 'Rate', 'correlation'])
r2_df.to_csv('result/test4/IDW2/r2.csv', index=False)


#ARIMA
r2 = []
station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
depths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for station in station_list:
    for depth in depths:
        depth_str = f'Soil_VWC_03_{depth}'
        data = pd.read_csv(f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv")
        df = pd.DataFrame(data)
        df['mask'] = df[depth_str].isna().astype(int)
        gapfilled_df = pd.read_csv(f"result/test4/ARIMA/ARIMA_{station}_gapfilling_{depth}.csv")
        mask_missing = df['mask'] == 1
        time = df.loc[mask_missing, 'Measure_Times'].values
        y_true_missing = df.loc[mask_missing, 'Soil_VWC_03'].values
        y_pred_missing = gapfilled_df.loc[mask_missing, depth_str].values
        result_df = pd.DataFrame({
            'Measure_Times': time,
            'y_true_missing': y_true_missing,
            'y_pred_missing': y_pred_missing
        })
        correlation = result_df['y_true_missing'].corr(result_df['y_pred_missing'])
        r2.append([station, depth, correlation])
        result_df.to_csv(f'result/test4/ARIMA2/{station}_r2_{depth}.csv', index=False)
r2_df = pd.DataFrame(r2, columns=['Station', 'Rate', 'correlation'])
r2_df.to_csv('result/test4/ARIMA2/r2.csv', index=False)


#LSTM
r2 = []
station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
depths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for station in station_list:
    for depth in depths:
        depth_str = f'Soil_VWC_03_{depth}'
        data = pd.read_csv(
            f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv")
        df = pd.DataFrame(data)
        df['mask'] = df[depth_str].isna().astype(int)
        gapfilled_df = pd.read_csv(f"result/test4/LSTM/{station}_gapfilling_{depth}.csv")
        mask_missing = df['mask'] == 1
        time = df.loc[mask_missing, 'Measure_Times'].values
        y_true_missing = df.loc[mask_missing, 'Soil_VWC_03'].values
        y_pred_missing = gapfilled_df.loc[mask_missing, depth_str].values
        result_df = pd.DataFrame({
            'Measure_Times': time,
            'y_true_missing': y_true_missing,
            'y_pred_missing': y_pred_missing
        })
        correlation = result_df['y_true_missing'].corr(result_df['y_pred_missing'])
        r2.append([station, depth, correlation])
        result_df.to_csv(f'result/test4/LSTM2/{station}_r2_{depth}.csv', index=False)
r2_df = pd.DataFrame(r2, columns=['Station', 'Rate', 'correlation'])
r2_df.to_csv('result/test4/LSTM2/r2.csv', index=False)


#SVR
r2 = []
station_list = ['L1', 'L2', 'L4', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L14']
depths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for station in station_list:
    for depth in depths:
        depth_str = f'Soil_VWC_03_{depth}'
        data = pd.read_csv(
            f"E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv")
        df = pd.DataFrame(data)
        df['mask'] = df[depth_str].isna().astype(int)
        gapfilled_df = pd.read_csv(f"result/test4/SVR/SVR_{station}_gapfilling_{depth}.csv")
        mask_missing = df['mask'] == 1
        time = df.loc[mask_missing, 'Measure_Times'].values
        y_true_missing = df.loc[mask_missing, 'Soil_VWC_03'].values
        y_pred_missing = gapfilled_df.loc[mask_missing, depth_str].values
        result_df = pd.DataFrame({
            'Measure_Times': time,
            'y_true_missing': y_true_missing,
            'y_pred_missing': y_pred_missing
        })
        correlation = result_df['y_true_missing'].corr(result_df['y_pred_missing'])
        r2.append([station, depth, correlation])
        result_df.to_csv(f'result/test4/SVR2/{station}_r2_{depth}.csv', index=False)
r2_df = pd.DataFrame(r2, columns=['Station', 'Rate', 'correlation'])
r2_df.to_csv('result/test4/SVR2/r2.csv', index=False)







