from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .views import register,trending_news,stock_alert_page,check_stock_alerts,add_ticker,StockDataView,stock_on_dashboard_data
from django.conf.urls.static import static
from .views import get_stock_data
from django.conf import settings

urlpatterns = [
    path('', views.index, name='home'), 
    path('login/', auth_views.LoginView.as_view(template_name='predict/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    
    path('predict/', views.predict, name='predict'),
    path('result/', views.result, name='result'),
    path('register/', views.register, name='register'),
    path('about/',views.about,name="about"),
    path('service/',views.service,name="service"),
    path('trending-news/', views.trending_news, name='trending_news'),
    path('check-stock-alerts/', views.stock_alert_page, name='stock_alert_page'),
    path('check-stock-alerts/action/', views.check_stock_alerts, name='check_stock_alerts'),
    path('add-ticker/', views.add_ticker, name='add_ticker'),
    path('profile/', views.user_profile, name='user_profile'),
    path('portfolio/', views.portfolio_view, name='portfolio'),
    path('get_stock_data/<str:symbol>/', StockDataView.as_view(), name='get_stock_data'),
    path('get_stock_data/', get_stock_data, name='get_stock_data'),
    path('get_stock_data/', stock_on_dashboard_data, name='stock_on_dashboard_data'),
    path('predict/real-time-price/', views.real_time_price, name='real_time_price_no_ticker'),
    path('predict/real-time-price/<str:ticker>/', views.real_time_price, name='real_time_price'),
    path('compare-stocks/', views.compare_stocks, name='compare_stocks'),
    path('dashboard/', views.stock_dashboard, name='dashboard'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
