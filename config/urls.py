# config/urls.py
from django.urls import path, include

urlpatterns = [
    path('api/', include('network.urls')),
]
