
from django.urls import path
from api import views
from django.conf import settings
from django.conf.urls.static import static
import django.contrib.auth.views as auth_views
from rest_framework_simplejwt.views import TokenRefreshView
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.utils.decorators import method_decorator

@ensure_csrf_cookie
def get_csrf_token(request):
    return JsonResponse({"csrfToken": request.META.get("CSRF_COOKIE", "")})


urlpatterns = [
    path("token/", views.MyTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("register/", views.RegisterView.as_view(), name="auth_register"),
    path("dashboard/", views.testEndPoint, name="test"),
    path('',views.getRoutes),
    path("profile/", views.get_profile, name="get_profile"),
    path("profile/update", views.update_profile, name="update_profile"),
    path("profile/change-password/", views.change_password, name="change_password"),
    path('api/password-reset/', views.PasswordResetRequestView.as_view(), name='password_reset_request'),
     path('password-reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
     path('upload/', views.upload_customers_from_csv, name='upload-customer'),
    path('customers/', views.get_customers, name='get-customers'),
     path('chatbot/', views.chatbot, name='chatbot'),
     path("api/csrf/", get_csrf_token, name="csrf-token"),
      path('api/chatbot/customers/', views.chatbot_customers, name="chatbot-customers"),
    path('upload_csv/', views.upload_and_predict, name='upload_csv'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)