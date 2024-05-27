from django.shortcuts import render
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def index(request):
    return render(request, 'index.html')
#입력된날짜,시간
def predict(request):
    datetime_input = request.GET.get('datetime_input')  # 'yyyy-mm-dd HH:MM'
    # date_A = request.GET.get('date_A')
    # time_C = request.GET.get('time_C')
#모델로드
    model = load_model('mini_app/models/mini_model.joblib')

#전처리 데이터로드
    df = pd.read_csv('mini_app/data/mini_data.csv')
    df['Datetime'] = pd.to_datetime(df['날짜'])
    df.set_index('Datetime', inplace=True)
    # df['Date'] = pd.to_datetime(df['Date'])
    # df['Time'] = pd.to_datetime(df['Time']).dt.time
    # df.set_index(['Date', 'Location', 'Time'], inplace=True)

#입력된날자,지역,시간에해당하는 데이터찾기

    try:
        input_data = df.loc[pd.to_datetime(datetime_input)]
        # input_data = df.loc[(pd.to_datetime(date_A), location_B, pd.to_datetime(time_C).time())]
    except KeyError:
        return render(request, 'predict.html', {'error': '데이터없수다'})

#데이터 전처리 < 넣어야하나?

    values = input_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values)

#예측을위한 데이터준비(최근 365일차)

    recent_data = scaled_data.reshape(1, 365, 1)

#예측수행

    prediction = model.predict(recent_data)
    probability = prediction[0][0]
    weather = '비' if probability > 0.5 else '맑음'

    # return render(request, 'predict.html', {'wether':weather, 'probability':probability,'date':date_A,'location':location_B,'time':time_C})
    return render(request, 'predict.html', {'weather': weather, 'probability': probability, 'datetime_input': datetime_input})