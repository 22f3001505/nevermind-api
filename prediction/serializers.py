from rest_framework import serializers
from .models import QuizAttempt, UserSkillProfile, CareerResult


class QuizAnswerSerializer(serializers.Serializer):
    """Validates a single quiz answer."""
    question_id = serializers.IntegerField(min_value=1, max_value=20)
    selected_option = serializers.IntegerField(min_value=0, max_value=3)


class QuizSubmissionSerializer(serializers.Serializer):
    """Validates the full quiz submission."""
    user_name = serializers.CharField(max_length=100, required=False, default="Anonymous")
    user_email = serializers.CharField(max_length=100, required=False, allow_blank=True, default="")
    session_id = serializers.CharField(max_length=64, required=False, allow_blank=True, default="")
    answers = QuizAnswerSerializer(many=True)

    def validate_answers(self, value):
        if len(value) < 10:
            raise serializers.ValidationError("At least 10 questions must be answered.")
        if len(value) > 20:
            raise serializers.ValidationError("Maximum 20 answers allowed.")

        q_ids = [a["question_id"] for a in value]
        if len(q_ids) != len(set(q_ids)):
            raise serializers.ValidationError("Duplicate question IDs found.")

        return value


class SkillProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSkillProfile
        fields = ["python", "javascript", "html_css", "sql", "problem_solving",
                  "ml_ai", "design", "networking", "devops", "communication"]


class CareerResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = CareerResult
        fields = ["results", "insight_text", "created_at"]


class QuizAttemptSerializer(serializers.ModelSerializer):
    skill_profile = SkillProfileSerializer(read_only=True)
    career_result = CareerResultSerializer(read_only=True)

    class Meta:
        model = QuizAttempt
        fields = ["id", "user_name", "session_id", "answers",
                  "skill_profile", "career_result", "created_at"]