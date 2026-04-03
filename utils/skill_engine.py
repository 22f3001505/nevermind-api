"""
Career Path AI — Skill Engine
Converts quiz answers into a normalized skill vector (0.0 - 1.0 per skill).
"""
import json
import os


# Load questions database
_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "../datasets/questions.json")
_questions_cache = None


def _load_questions():
    global _questions_cache
    if _questions_cache is None:
        with open(_QUESTIONS_PATH, "r") as f:
            _questions_cache = {q["id"]: q for q in json.load(f)}
    return _questions_cache


SKILL_DIMENSIONS = [
    "python", "javascript", "html_css", "sql", "problem_solving",
    "ml_ai", "design", "networking", "devops", "communication"
]


def calculate_skills(answers):
    """
    Convert quiz answers into a normalized skill vector.

    Args:
        answers: list of dicts, each with:
            - question_id (int): ID of the question
            - selected_option (int): Index of the selected option (0-3)

    Returns:
        dict: Skill vector mapping skill names to scores (0.0 - 1.0)
              e.g. {"python": 0.72, "javascript": 0.35, ...}
    """
    questions = _load_questions()
    skill_scores = {skill: 0.0 for skill in SKILL_DIMENSIONS}
    skill_counts = {skill: 0 for skill in SKILL_DIMENSIONS}

    for answer in answers:
        q_id = answer.get("question_id")
        option_idx = answer.get("selected_option", 0)

        question = questions.get(q_id)
        if not question:
            continue

        # Get the selected option's skill contributions
        options = question.get("options", [])
        if option_idx < 0 or option_idx >= len(options):
            continue

        selected = options[option_idx]
        skill_map = selected.get("skills", {})

        for skill, weight in skill_map.items():
            if skill in skill_scores:
                skill_scores[skill] += weight
                skill_counts[skill] += 1

    # Normalize each skill to 0.0 - 1.0 range
    # Max possible score per skill = sum of all questions that map to it
    for skill in SKILL_DIMENSIONS:
        # Calculate max possible score for this skill
        max_score = 0.0
        for q in questions.values():
            max_option_score = 0.0
            for option in q.get("options", []):
                opt_skills = option.get("skills", {})
                if skill in opt_skills:
                    max_option_score = max(max_option_score, opt_skills[skill])
            max_score += max_option_score

        if max_score > 0:
            skill_scores[skill] = round(min(skill_scores[skill] / max_score, 1.0), 3)
        else:
            skill_scores[skill] = 0.0

    return skill_scores


def get_skill_level(score):
    """Convert a numeric skill score to a human-readable level."""
    if score >= 0.75:
        return "Advanced"
    elif score >= 0.40:
        return "Intermediate"
    elif score >= 0.15:
        return "Beginner"
    else:
        return "Novice"


def get_skill_summary(skill_vector):
    """
    Generate a summary of skills with levels.

    Returns:
        list of dicts: [{"skill": str, "score": float, "level": str}, ...]
    """
    return [
        {
            "skill": skill,
            "score": score,
            "level": get_skill_level(score)
        }
        for skill, score in sorted(skill_vector.items(), key=lambda x: -x[1])
    ]
