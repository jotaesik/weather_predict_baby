from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import tensorflow as tf

# LSTM 모델을 불러오는 함수
def load_lstm_model():
    # 모델을 불러올 경로 설정
    model_path = '/home/encore/mini_project/mini_app/models/lstm_model.h5'
    # 모델 불러오기
    model = tf.keras.models.load_model(model_path)
    return model

# 날짜만 입력받아 예측을 수행하는 뷰 함수
def weather_prediction(request):
    if request.method == 'POST':
        # POST 요청에서 날짜 데이터 가져오기
        date_input = request.POST.get('date_input')
        
        # LSTM 모델 불러오기
        lstm_model = load_lstm_model()
        
        # 예측을 위한 데이터 준비 (전처리)
        processed_data = preprocess_data(date_input)
        
        # 예측 수행
        prediction = lstm_model.predict(processed_data)
        
        # 예측 결과를 사용자에게 보여주기
        # render 함수를 사용하여 predict.html을 렌더링하고, 예측 결과를 함께 전달
        return render(request, 'predict.html', {'prediction': prediction})
    else:
        # GET 요청 시 index.html 렌더링
        return render(request, 'index.html')

# 데이터 전처리 함수 (필요에 따라 구현)
def preprocess_data(date_input):
    # 날짜 데이터를 모델 입력 형식에 맞게 전처리
    # 예시: 날짜 데이터를 숫자 또는 특정 형식으로 변환하여 모델에 입력할 수 있는 형태로 가공
    processed_data = ...
    return processed_data
