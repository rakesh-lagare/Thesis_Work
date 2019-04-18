
from django.contrib import admin
from django.urls import path
from pages import views

from django.contrib import admin
from django.urls import path



urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', views.home,  name="home"),
    path('images/', views.images_display,  name="images"),
    path('brush_graph/', views.brush_graph,  name="brush_graph"),
    path('csv_upload/', views.csv_upload,  name="csv_upload"),
]
