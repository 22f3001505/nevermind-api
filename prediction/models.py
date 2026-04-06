from django.db import models


class QuizAttempt(models.Model):
    """Records a single quiz attempt."""
    user_name = models.CharField(max_length=100, default="Anonymous")
    user_email = models.CharField(max_length=100, blank=True, default="")
    session_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    answers = models.JSONField(help_text="List of {question_id, selected_option}")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["session_id", "-created_at"]),
        ]

    def __str__(self):
        return f"Quiz by {self.user_name} at {self.created_at}"


class UserSkillProfile(models.Model):
    """Stores the computed skill vector from a quiz attempt."""
    quiz_attempt = models.OneToOneField(QuizAttempt, on_delete=models.CASCADE, related_name="skill_profile")
    python = models.FloatField(default=0.0)
    javascript = models.FloatField(default=0.0)
    html_css = models.FloatField(default=0.0)
    sql = models.FloatField(default=0.0)
    problem_solving = models.FloatField(default=0.0)
    ml_ai = models.FloatField(default=0.0)
    design = models.FloatField(default=0.0)
    networking = models.FloatField(default=0.0)
    devops = models.FloatField(default=0.0)
    communication = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def to_dict(self):
        return {
            "python": self.python,
            "javascript": self.javascript,
            "html_css": self.html_css,
            "sql": self.sql,
            "problem_solving": self.problem_solving,
            "ml_ai": self.ml_ai,
            "design": self.design,
            "networking": self.networking,
            "devops": self.devops,
            "communication": self.communication,
        }

    def __str__(self):
        return f"Skills for Quiz #{self.quiz_attempt_id}"


class CareerResult(models.Model):
    """Stores career prediction results from a quiz attempt."""
    quiz_attempt = models.OneToOneField(QuizAttempt, on_delete=models.CASCADE, related_name="career_result")
    top_career = models.CharField(max_length=100, blank=True, default="")
    confidence = models.FloatField(default=0.0)
    results = models.JSONField(help_text="List of {career, confidence, rank}")
    insight_text = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["top_career"]),
        ]

    def __str__(self):
        return f"Results for Quiz #{self.quiz_attempt_id}"


class TopicProgress(models.Model):
    """Tracks user progress on roadmap topics (checkboxes)."""
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE, related_name="progress")
    career = models.CharField(max_length=100, db_index=True)
    level = models.CharField(max_length=20)  # beginner, intermediate, advanced
    topic_index = models.IntegerField()  # 0-based index within level
    completed = models.BooleanField(default=True)
    completed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "career", "level", "topic_index")
        indexes = [
            models.Index(fields=["user", "career"]),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.career} [{self.level}][{self.topic_index}]"