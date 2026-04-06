from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import (
    QuizQuestionsView, QuizSubmitView,
    CareersListView, CareerRoadmapView,
    HealthCheckView, QuizHistoryView, PlatformStatsView,
    SignupView, UserProfileView, TopicProgressView
)

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('stats/', PlatformStatsView.as_view(), name='platform-stats'),
    path('quiz/questions/', QuizQuestionsView.as_view(), name='quiz-questions'),
    path('quiz/submit/', QuizSubmitView.as_view(), name='quiz-submit'),
    path('quiz/history/', QuizHistoryView.as_view(), name='quiz-history'),
    path('careers/', CareersListView.as_view(), name='careers-list'),
    path('careers/<str:career_name>/roadmap/', CareerRoadmapView.as_view(), name='career-roadmap'),

    # Auth
    path('auth/signup/', SignupView.as_view(), name='auth-signup'),
    path('auth/login/', TokenObtainPairView.as_view(), name='auth-login'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='auth-refresh'),
    path('auth/profile/', UserProfileView.as_view(), name='auth-profile'),

    # Progress tracking
    path('progress/<str:career_slug>/', TopicProgressView.as_view(), name='topic-progress'),
]
