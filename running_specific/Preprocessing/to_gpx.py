from fit2gpx import Converter

conv = Converter()

df_lap, df_point = conv.fit_to_dataframes(fname='12966366635.fit')

start_date = df_point['timestamp'].min()
average_hr = df_point["heart_rate"].mean().round(1)
max_hr = df_point["heart_rate"].max().round(1)
temp = df_point["temperature"].round(1)

print(start_date, average_hr, max_hr, temp)