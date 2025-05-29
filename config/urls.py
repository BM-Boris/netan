# config/urls.py
from django.contrib import admin
from django.urls import path, include
from network.views import BuildNetworkView

urlpatterns = [
    
    path('api/build-network/', BuildNetworkView.as_view(), name='build-network'),
    #path('api/', include('network.urls')),
    path('admin/', admin.site.urls),
]
