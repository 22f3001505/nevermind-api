from django.urls import path
from .views import (
    QuizQuestionsView, QuizSubmitView,
    CareersListView, CareerRoadmapView,
    HealthCheckView, QuizHistoryView, PlatformStatsView
)

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('stats/', PlatformStatsView.as_view(), name='platform-stats'),
    path('quiz/questions/', QuizQuestionsView.as_view(), name='quiz-questions'),
    path('quiz/submit/', QuizSubmitView.as_view(), name='quiz-submit'),
    path('quiz/history/', QuizHistoryView.as_view(), name='quiz-history'),
    path('careers/', CareersListView.as_view(), name='careers-list'),
    path('careers/<str:career_name>/roadmap/', CareerRoadmapView.as_view(), name='career-roadmap'),
]
