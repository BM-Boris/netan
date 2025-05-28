# network/urls.py
from django.urls import path
from .views import BuildNetworkView

urlpatterns = [
    path('build-network/', BuildNetworkView.as_view(), name='build-network'),
]
