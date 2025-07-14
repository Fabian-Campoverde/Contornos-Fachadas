from django.urls import path
from .views import procesar_imagen


urlpatterns = [
     path('', procesar_imagen, name='index'),
]