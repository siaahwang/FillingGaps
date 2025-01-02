
# 定义站点列表
station_list = ['L3', 'M3', 'M4', 'M7', 'M8', 'L5', 'L7', 'L13']
#missing_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for station in station_list:
    df = pd.read_csv(f'F:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\30min_2019-2020\\{station}_SM.csv')
    df2 = pd.read_csv(
        f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\data_missing_{int(rate * 100)}.csv')
    df[f'Soil_VWC_03_{rate}'] = df2[f'{station}']


    df.to_csv(f'E:\\SM\\interpolation\\dataset\\SMN-SDR_ground-data(03cm)_2018-2020\\test4\\{station}_SM.csv',index=False)