# api/urls.py
from django.urls import path
from . import views
from .views import SignupView, LoginView, ProfileView

urlpatterns = [
     path("", views.dashboard, name="dashboard"),
    #  upande wa ku logout 
     path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('predict-eta/', views.predict_eta, name='predict_eta'),
    path('predict-eta-smart/', views.predict_eta_smart, name='predict_eta_smart'),
    path('track-orders/', views.track_orders, name='track_orders'),
    path('optimize-route/', views.optimize_route, name='optimize_route'),
    path('carbon-footprint/', views.carbon_footprint, name='carbon_footprint_page'),
    path('courier-performance/', views.courier_performance, name='courier_performance'),
    path('system-insights/', views.system_insights_page, name='system_insights_page'),
    path('predict-eta-page/', views.predict_eta_page, name='predict_eta_page'),  # HTML page
    path("route-search/", views.route_search_page, name="route_search_page"),
     # API endpointa
    path("api/detect_anomalies/", views.detect_anomalies, name="detect_anomalies_api"),
    path("track-orders-ml/", views.track_orders_ml, name="track_orders_ml"),


    # HTML page
    path("detect_anomalies/", views.detect_anomalies_page, name="detect_anomalies_page"),

    # hapa ni apia zangu 
path('system-insights-page/', views.system_insights, name='system_insights_page'),
path('predict-eta/', views.predict_eta, name='predict_eta'),
path('track-orders/', views.track_orders, name='track_orders'),
path('carbon-footprint/', views.carbon_footprint, name='carbon_footprint'),
path('system-insights/', views.system_insights, name='system_insights'),
path('route-search/', views.route_search, name='route_search'),

]