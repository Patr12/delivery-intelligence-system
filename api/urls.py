# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
     path("", views.dashboard, name="dashboard"),
    path('predict-eta/', views.predict_eta, name='predict_eta'),
    path('predict-eta-smart/', views.predict_eta_smart, name='predict_eta_smart'),
    path('track-orders/', views.track_orders, name='track_orders'),
    path('optimize-route/', views.optimize_route, name='optimize_route'),
    path('carbon-footprint/', views.carbon_footprint, name='carbon_footprint'),
    path('courier-performance/', views.courier_performance, name='courier_performance'),
    path('system-insights/', views.system_insights_page, name='system_insights_page'),
    path('predict-eta-page/', views.predict_eta_page, name='predict_eta_page'),  # HTML page
    path("route-search/", views.route_search_page, name="route_search"),
     # API endpoint
    path("api/detect_anomalies/", views.detect_anomalies, name="detect_anomalies_api"),

    # HTML page
    path("detect_anomalies/", views.detect_anomalies_page, name="detect_anomalies_page"),

    path('system-insights/', views.system_insights, name='system_insights'),
]