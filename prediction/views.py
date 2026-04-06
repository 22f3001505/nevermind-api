"""
Never Mind — API Views (Production)
Full error handling, health checks, quiz history, platform stats.
Auth: signup, JWT login, progress tracking.
"""
import json
import os
import logging
import traceback

from django.db.models import Count, Avg
from django.utils import timezone
from django.contrib.auth.models import User

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny

from .models import QuizAttempt, UserSkillProfile, CareerResult, TopicProgress
from .serializers import QuizSubmissionSerializer, QuizAttemptSerializer

from utils.skill_engine import calculate_skills, get_skill_summary
from utils.rule_engine import generate_insight, refine_predictions
from utils.roadmap_engine import get_roadmap, get_all_careers, get_personalized_roadmap

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml_models.predict import predict_careers, predict_all_careers

logger = logging.getLogger('prediction')


# ═══════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════

class HealthCheckView(APIView):
    """GET /api/health/ — System health check for connectivity verification."""

    def get(self, request):
        try:
            from ml_models.predict import get_career_list, get_skill_columns
            careers = get_career_list()
            skills = get_skill_columns()

            quiz_count = QuizAttempt.objects.count()

            return Response({
                "status": "healthy",
                "engine": "Never Mind v2.0",
                "model": "Ensemble (RF+GB)",
                "careers": len(careers),
                "skills": len(skills),
                "total_quizzes": quiz_count,
                "timestamp": timezone.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return Response({
                "status": "degraded",
                "error": str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# ═══════════════════════════════════════════════════════════
# Quiz Questions
# ═══════════════════════════════════════════════════════════

class QuizQuestionsView(APIView):
    """GET /api/quiz/questions/ — Return all quiz questions."""

    def get(self, request):
        try:
            questions_path = os.path.join(
                os.path.dirname(__file__), "../datasets/questions.json"
            )
            with open(questions_path, "r") as f:
                questions = json.load(f)

            sanitized = []
            for q in questions:
                sanitized.append({
                    "id": q["id"],
                    "text": q["text"],
                    "category": q["category"],
                    "options": [
                        {"label": opt["label"], "index": i}
                        for i, opt in enumerate(q["options"])
                    ]
                })

            return Response({"questions": sanitized, "total": len(sanitized)})

        except FileNotFoundError:
            logger.error("questions.json not found")
            return Response(
                {"error": "Quiz data not found"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return Response(
                {"error": "Failed to load questions"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ═══════════════════════════════════════════════════════════
# Quiz Submission (Core Pipeline)
# ═══════════════════════════════════════════════════════════

class QuizSubmitView(APIView):
    """POST /api/quiz/submit/ — Submit quiz answers → career prediction + roadmap."""

    def post(self, request):
        serializer = QuizSubmissionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        answers = data["answers"]

        try:
            # Step 1: Calculate Skill Vector
            skill_vector = calculate_skills(answers)
            skill_summary = get_skill_summary(skill_vector)
            logger.info(f"Skill vector computed: {skill_vector}")

            # Step 2: ML Prediction
            top_careers = predict_careers(skill_vector, top_n=3)
            all_careers = predict_all_careers(skill_vector)
            logger.info(f"Prediction: {[c['career'] for c in top_careers]}")

            # Step 3: Rule Engine Refinement
            top_careers = refine_predictions(skill_vector, top_careers)

            # Step 4: Generate AI Insight
            top_career_name = top_careers[0]["career"]
            insight = generate_insight(skill_vector, top_career_name)

            # Step 5: Get Roadmap
            roadmap = get_personalized_roadmap(top_career_name, skill_vector)

            # Step 6: Save to Database
            quiz_attempt = QuizAttempt.objects.create(
                user_name=data.get("user_name", "Anonymous"),
                user_email=data.get("user_email", ""),
                session_id=data.get("session_id", ""),
                answers=answers
            )

            UserSkillProfile.objects.create(
                quiz_attempt=quiz_attempt,
                **skill_vector
            )

            CareerResult.objects.create(
                quiz_attempt=quiz_attempt,
                top_career=top_career_name,
                confidence=top_careers[0]["confidence"],
                results=top_careers,
                insight_text=insight
            )

            logger.info(f"Quiz #{quiz_attempt.id} saved → {top_career_name}")

            # Step 7: Build Response
            return Response({
                "quiz_id": quiz_attempt.id,
                "skills": skill_vector,
                "skill_summary": skill_summary,
                "top_careers": top_careers,
                "all_careers": all_careers,
                "insight": insight,
                "roadmap": roadmap
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Quiz submission error: {traceback.format_exc()}")
            return Response(
                {"error": "Analysis failed. Please try again.", "detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ═══════════════════════════════════════════════════════════
# Careers
# ═══════════════════════════════════════════════════════════

class CareersListView(APIView):
    """GET /api/careers/ — List all 15 career paths with metadata."""

    def get(self, request):
        try:
            careers = get_all_careers()
            return Response({"careers": careers, "total": len(careers)})
        except Exception as e:
            logger.error(f"Careers list error: {e}")
            return Response(
                {"error": "Failed to load careers"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CareerRoadmapView(APIView):
    """GET /api/careers/<career_name>/roadmap/ — Get detailed roadmap."""

    def _normalize(self, name):
        """Normalize career name for matching: 'UI-UX Designer' → 'ui/ux designer'"""
        import re
        n = name.replace("-", " ").replace("_", " ").strip().lower()
        # "ui ux" → "ui/ux", "qa  test" → "qa / test"
        return n

    def _fuzzy_match(self, input_name, career_name):
        """Check if input matches career, handling slash variations."""
        import re
        # Normalize both: remove spaces around /, lowercase
        def norm(s):
            s = s.lower().strip()
            s = re.sub(r'\s*/\s*', '/', s)  # "UI / UX" → "UI/UX"
            s = re.sub(r'\s+', ' ', s)
            return s
        # Also try without slash: "UI UX" should match "UI/UX"
        input_norm = norm(input_name)
        career_norm = norm(career_name)
        if input_norm == career_norm:
            return True
        # Try replacing spaces with slash: "UI UX" → "UI/UX"
        input_slash = re.sub(r'(\w)\s+(\w)', r'\1/\2', input_norm)
        if input_slash == career_norm:
            return True
        return False

    def get(self, request, career_name):
        try:
            career_name_clean = career_name.replace("-", " ").replace("_", " ")
            roadmap = get_roadmap(career_name_clean)

            if not roadmap:
                all_careers = get_all_careers()
                for career in all_careers:
                    if self._fuzzy_match(career_name_clean, career["name"]):
                        roadmap = get_roadmap(career["name"])
                        break

            if not roadmap:
                return Response(
                    {"error": f"Career '{career_name_clean}' not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            return Response(roadmap)

        except Exception as e:
            logger.error(f"Roadmap error for {career_name}: {e}")
            return Response(
                {"error": "Failed to load roadmap"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ═══════════════════════════════════════════════════════════
# Quiz History
# ═══════════════════════════════════════════════════════════

class QuizHistoryView(APIView):
    """GET /api/quiz/history/ — Get recent quiz attempts."""

    def get(self, request):
        try:
            session_id = request.headers.get("X-Session-Id", "")
            limit = min(int(request.query_params.get("limit", 10)), 50)

            queryset = QuizAttempt.objects.select_related(
                'skill_profile', 'career_result'
            ).order_by('-created_at')

            if session_id:
                queryset = queryset.filter(session_id=session_id)

            attempts = queryset[:limit]

            results = []
            for attempt in attempts:
                entry = {
                    "id": attempt.id,
                    "user_name": attempt.user_name,
                    "created_at": attempt.created_at.isoformat(),
                    "answers_count": len(attempt.answers) if attempt.answers else 0,
                }

                if hasattr(attempt, 'career_result') and attempt.career_result:
                    entry["top_career"] = attempt.career_result.top_career
                    entry["confidence"] = attempt.career_result.confidence
                    entry["results"] = attempt.career_result.results
                    entry["insight"] = attempt.career_result.insight_text

                if hasattr(attempt, 'skill_profile') and attempt.skill_profile:
                    entry["skills"] = attempt.skill_profile.to_dict()

                results.append(entry)

            return Response({
                "history": results,
                "total": len(results)
            })

        except Exception as e:
            logger.error(f"History error: {e}")
            return Response(
                {"error": "Failed to load history"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ═══════════════════════════════════════════════════════════
# Platform Stats
# ═══════════════════════════════════════════════════════════

class PlatformStatsView(APIView):
    """GET /api/stats/ — Aggregate platform statistics."""

    def get(self, request):
        try:
            total_quizzes = QuizAttempt.objects.count()

            # Top careers by frequency
            top_careers = list(
                CareerResult.objects.values('top_career')
                .annotate(count=Count('id'))
                .order_by('-count')[:5]
            )

            # Average skill scores across all users
            avg_skills = UserSkillProfile.objects.aggregate(
                python=Avg('python'),
                javascript=Avg('javascript'),
                html_css=Avg('html_css'),
                sql=Avg('sql'),
                problem_solving=Avg('problem_solving'),
                ml_ai=Avg('ml_ai'),
                design=Avg('design'),
                networking=Avg('networking'),
                devops=Avg('devops'),
                communication=Avg('communication'),
            )

            # Clean None values
            avg_skills = {k: round(v, 3) if v else 0.0 for k, v in avg_skills.items()}

            return Response({
                "total_quizzes": total_quizzes,
                "top_careers": top_careers,
                "avg_skills": avg_skills,
                "careers_available": 15,
                "questions_count": 20,
                "model_type": "Ensemble (RF+GB)",
            })

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return Response(
                {"error": "Failed to load stats"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SignupView(APIView):
    """POST /api/auth/signup/ — Register a new user."""
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username", "").strip()
        email = request.data.get("email", "").strip()
        password = request.data.get("password", "")

        if not username or not password:
            return Response(
                {"error": "Username and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(password) < 6:
            return Response(
                {"error": "Password must be at least 6 characters"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "Username already taken"},
                status=status.HTTP_409_CONFLICT
            )

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        # Generate JWT tokens
        from rest_framework_simplejwt.tokens import RefreshToken
        refresh = RefreshToken.for_user(user)

        return Response({
            "message": "Account created successfully",
            "user": {"id": user.id, "username": user.username, "email": user.email},
            "tokens": {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            }
        }, status=status.HTTP_201_CREATED)


class UserProfileView(APIView):
    """GET /api/auth/profile/ — Get current user info."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        progress_count = TopicProgress.objects.filter(user=user).count()
        return Response({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "completed_topics": progress_count,
        })


class TopicProgressView(APIView):
    """
    GET  /api/progress/<career_slug>/ — Get completed topics for a career.
    POST /api/progress/<career_slug>/ — Toggle a topic completion.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, career_slug):
        career = career_slug.replace('-', ' ').replace('  ', ' ')
        items = TopicProgress.objects.filter(
            user=request.user, career__iexact=career
        ).values_list('level', 'topic_index')

        # Also try fuzzy match
        if not items.exists():
            import re
            all_progress = TopicProgress.objects.filter(user=request.user)
            for p in all_progress:
                norm_p = re.sub(r'[\s/\-]+', ' ', p.career).strip().lower()
                norm_c = re.sub(r'[\s/\-]+', ' ', career).strip().lower()
                if norm_p == norm_c:
                    career = p.career
                    items = TopicProgress.objects.filter(
                        user=request.user, career=career
                    ).values_list('level', 'topic_index')
                    break

        completed = [f"{level}_{idx}" for level, idx in items]
        return Response({"career": career, "completed": completed})

    def post(self, request, career_slug):
        career = career_slug.replace('-', ' ').replace('  ', ' ')
        level = request.data.get('level')
        topic_index = request.data.get('topic_index')
        completed = request.data.get('completed', True)

        if level is None or topic_index is None:
            return Response(
                {"error": "level and topic_index are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Resolve actual career name from roadmaps
        from utils.roadmap_engine import get_roadmap, get_all_careers
        import re
        actual_career = career
        for c in get_all_careers():
            norm_c = re.sub(r'[\s/\-]+', ' ', c['name']).strip().lower()
            norm_input = re.sub(r'[\s/\-]+', ' ', career).strip().lower()
            if norm_c == norm_input:
                actual_career = c['name']
                break

        if completed:
            TopicProgress.objects.update_or_create(
                user=request.user,
                career=actual_career,
                level=level,
                topic_index=topic_index,
                defaults={'completed': True}
            )
        else:
            TopicProgress.objects.filter(
                user=request.user,
                career=actual_career,
                level=level,
                topic_index=topic_index
            ).delete()

        # Return updated progress
        items = TopicProgress.objects.filter(
            user=request.user, career=actual_career
        ).values_list('level', 'topic_index')
        completed_list = [f"{l}_{i}" for l, i in items]

        return Response({
            "career": actual_career,
            "completed": completed_list,
            "total_completed": len(completed_list)
        })