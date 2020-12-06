from django.urls import path
from tensor import views

urlpatterns = [
    path("", views.index, name="index"),
    path('api', views.detail, name="detail"),
    path('api/obj_list', views.load_obj_list, name="obj_list"),
    path('api/train', views.make_model_train, name="make_model_train"),
]