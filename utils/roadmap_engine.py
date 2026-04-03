"""
Career Path AI — Roadmap Engine
Loads career roadmaps and generates personalized learning paths based on skill gaps.
"""
import json
import os


_ROADMAPS_PATH = os.path.join(os.path.dirname(__file__), "../datasets/roadmaps.json")
_roadmaps_cache = None


def _load_roadmaps():
    global _roadmaps_cache
    if _roadmaps_cache is None:
        with open(_ROADMAPS_PATH, "r") as f:
            _roadmaps_cache = json.load(f)
    return _roadmaps_cache


def get_roadmap(career_name):
    """
    Get the complete roadmap for a career.
    Supports fuzzy matching for slash variations:
      "UI UX Designer" → "UI/UX Designer"
      "QA  Test Engineer" → "QA / Test Engineer"
    """
    import re
    roadmaps = _load_roadmaps()

    # Try exact match first
    if career_name in roadmaps:
        return roadmaps[career_name]

    # Fuzzy match: strip all slashes, collapse spaces, lowercase
    def norm(s):
        s = s.replace('/', ' ').replace('-', ' ').replace('_', ' ')
        s = re.sub(r'\s+', ' ', s).strip().lower()
        return s

    input_norm = norm(career_name)
    for key in roadmaps:
        if norm(key) == input_norm:
            return roadmaps[key]

    return None


def get_all_careers():
    """
    Get metadata for all careers (without detailed roadmaps).

    Returns:
        list of dicts: [{name, description, avg_salary, growth_outlook, key_skills}, ...]
    """
    roadmaps = _load_roadmaps()
    careers = []
    for name, data in roadmaps.items():
        careers.append({
            "name": name,
            "description": data.get("description", ""),
            "avg_salary": data.get("avg_salary", ""),
            "growth_outlook": data.get("growth_outlook", ""),
            "key_skills": data.get("key_skills", []),
            "key_companies": data.get("key_companies", [])
        })
    return careers


def get_personalized_roadmap(career_name, skill_vector):
    """
    Generate a personalized roadmap that highlights skill gaps.

    Args:
        career_name: str, target career
        skill_vector: dict of current skill scores

    Returns:
        dict with roadmap + personalized notes based on skill gaps
    """
    career_data = get_roadmap(career_name)
    if not career_data:
        return None

    # Skill requirements per career (approximate mappings)
    career_skill_requirements = {
        "Frontend Developer": {"javascript": 0.8, "html_css": 0.8, "design": 0.6},
        "Backend Developer": {"python": 0.8, "sql": 0.7, "problem_solving": 0.7},
        "Full Stack Developer": {"python": 0.6, "javascript": 0.7, "html_css": 0.6, "sql": 0.6},
        "Mobile App Developer": {"javascript": 0.7, "design": 0.5, "problem_solving": 0.6},
        "DevOps Engineer": {"devops": 0.8, "networking": 0.6, "python": 0.5},
        "Cloud Engineer": {"devops": 0.7, "networking": 0.7, "python": 0.5},
        "Data Analyst": {"sql": 0.8, "python": 0.5, "problem_solving": 0.6},
        "Data Scientist": {"python": 0.8, "ml_ai": 0.8, "problem_solving": 0.7},
        "ML Engineer": {"python": 0.9, "ml_ai": 0.9, "problem_solving": 0.8},
        "UI/UX Designer": {"design": 0.9, "html_css": 0.6, "communication": 0.7},
        "Cybersecurity Analyst": {"networking": 0.8, "problem_solving": 0.7, "python": 0.5},
        "QA / Test Engineer": {"problem_solving": 0.7, "python": 0.4, "communication": 0.6},
        "Database Administrator": {"sql": 0.9, "networking": 0.4, "problem_solving": 0.6},
        "System Administrator": {"networking": 0.7, "devops": 0.6, "python": 0.4},
        "Network Engineer": {"networking": 0.9, "problem_solving": 0.6, "devops": 0.4}
    }

    requirements = career_skill_requirements.get(career_name, {})

    # Calculate gaps
    skill_gaps = []
    for skill, required in requirements.items():
        current = skill_vector.get(skill, 0.0)
        gap = max(0, required - current)
        if gap > 0:
            skill_gaps.append({
                "skill": skill,
                "current": round(current, 2),
                "required": required,
                "gap": round(gap, 2)
            })

    skill_gaps.sort(key=lambda x: -x["gap"])

    # Determine starting level
    avg_gap = sum(g["gap"] for g in skill_gaps) / max(len(skill_gaps), 1)
    if avg_gap > 0.5:
        suggested_start = "beginner"
    elif avg_gap > 0.2:
        suggested_start = "intermediate"
    else:
        suggested_start = "advanced"

    return {
        "career": career_name,
        "description": career_data.get("description", ""),
        "avg_salary": career_data.get("avg_salary", ""),
        "growth_outlook": career_data.get("growth_outlook", ""),
        "key_skills": career_data.get("key_skills", []),
        "key_companies": career_data.get("key_companies", []),
        "roadmap": career_data.get("roadmap", {}),
        "skill_gaps": skill_gaps,
        "suggested_start_level": suggested_start,
        "total_estimated_weeks": sum(
            int(item.get("duration", "0 weeks").split()[0])
            for level in career_data.get("roadmap", {}).values()
            for item in level
        )
    }
