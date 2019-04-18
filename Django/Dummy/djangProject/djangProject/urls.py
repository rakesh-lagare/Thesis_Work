from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('about/',views.about),
    path('home/',views.home),
    path('test_final/',views.test_final),

]
