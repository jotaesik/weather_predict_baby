from django.urls import path
from . import views

urlpatterns = [
    path('', views.weather_prediction, name='weather_prediction'),
]
