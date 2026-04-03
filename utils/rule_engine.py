"""
Career Path AI — Rule Engine
Post-ML refinement rules and AI insight generation.
Provides explainable, human-readable career explanations.
"""


def generate_insight(skill_vector, top_career):
    """
    Generate a human-readable AI insight explaining WHY this career was recommended.

    Args:
        skill_vector: dict of skill scores (0.0-1.0)
        top_career: str name of the top predicted career

    Returns:
        str: Insight text explaining the recommendation
    """
    # Find top 3 strongest skills
    sorted_skills = sorted(skill_vector.items(), key=lambda x: -x[1])
    top_skills = sorted_skills[:3]
    weak_skills = [s for s in sorted_skills if s[1] < 0.25]

    # Build skill description
    skill_names = {
        "python": "Python programming",
        "javascript": "JavaScript development",
        "html_css": "HTML/CSS and web design",
        "sql": "database and SQL",
        "problem_solving": "analytical and problem-solving",
        "ml_ai": "machine learning and AI",
        "design": "creative design and UX",
        "networking": "networking and infrastructure",
        "devops": "DevOps and automation",
        "communication": "communication and teamwork"
    }

    # Career-specific insights
    career_insights = {
        "Frontend Developer": "Your strong frontend and design skills make you a natural fit for building engaging user interfaces. Frontend developers are in high demand, with the React ecosystem being particularly lucrative.",
        "Backend Developer": "Your backend strength combined with strong problem-solving skills positions you well for building robust server-side systems. Backend developers are the backbone of every tech company.",
        "Full Stack Developer": "Your balanced skill set across frontend and backend technologies makes you versatile. Full Stack developers are highly valued in startups and mid-size companies for their ability to work across the entire stack.",
        "Mobile App Developer": "Your mobile development aptitude combined with UI skills aligns perfectly with the growing mobile-first world. Mobile developers are in high demand as app usage continues to surge.",
        "DevOps Engineer": "Your infrastructure and automation skills indicate a strong fit for DevOps. This role bridges development and operations, and is one of the highest-paying specializations in tech.",
        "Cloud Engineer": "Your cloud and infrastructure expertise positions you for the rapidly growing cloud computing market. Cloud engineers are critical as organizations accelerate their digital transformation.",
        "Data Analyst": "Your data handling and analytical skills make you ideal for turning raw data into actionable business insights. Data-driven decision making is now essential across all industries.",
        "Data Scientist": "Your strong ML/AI and Python skills combined with analytical thinking make you a prime candidate for data science. This remains one of the most impactful roles in modern tech.",
        "ML Engineer": "Your deep ML/AI expertise and strong programming skills position you for the cutting edge of AI engineering. ML Engineers build the systems that power intelligent applications.",
        "UI/UX Designer": "Your exceptional design sense and user empathy make you a natural UX professional. Great design is a key competitive advantage, making this role increasingly strategic.",
        "Cybersecurity Analyst": "Your networking and security awareness skills are critical in today's threat landscape. Cybersecurity professionals are in extreme demand with a near-zero unemployment rate.",
        "QA / Test Engineer": "Your attention to quality and systematic thinking make you well-suited for ensuring software reliability. QA engineers are essential for shipping high-quality products.",
        "Database Administrator": "Your strong database skills and systematic approach make you ideal for managing the data infrastructure that every application depends on.",
        "System Administrator": "Your infrastructure management skills and system-level thinking position you well for keeping organizations' IT systems running smoothly and securely.",
        "Network Engineer": "Your deep networking knowledge makes you essential for designing and maintaining the communication infrastructure that connects everything."
    }

    # Compose insight
    top_skill_text = ", ".join([skill_names.get(s[0], s[0]) for s in top_skills[:2]])
    career_text = career_insights.get(top_career, f"Your skill profile aligns well with the {top_career} career path.")

    insight = f"Based on our analysis, your strongest areas are {top_skill_text}. {career_text}"

    # Add growth suggestion if there are weak areas
    if weak_skills:
        weak_name = skill_names.get(weak_skills[0][0], weak_skills[0][0])
        insight += f" To further strengthen your profile, consider developing your {weak_name} skills."

    return insight


def refine_predictions(skill_vector, ml_predictions):
    """
    Apply rule-based refinements to ML predictions.
    Adjusts scores based on strong skill correlations.

    Args:
        skill_vector: dict of skill scores
        ml_predictions: list of {"career": str, "confidence": float, "rank": int}

    Returns:
        list: Refined predictions (same format, potentially reordered)
    """
    rules = [
        # (condition_func, career_to_boost, boost_amount)
        (lambda s: s.get("design", 0) > 0.8 and s.get("html_css", 0) > 0.7,
         "UI/UX Designer", 0.05),
        (lambda s: s.get("ml_ai", 0) > 0.8 and s.get("python", 0) > 0.8,
         "ML Engineer", 0.05),
        (lambda s: s.get("networking", 0) > 0.85,
         "Network Engineer", 0.04),
        (lambda s: s.get("sql", 0) > 0.85 and s.get("python", 0) > 0.5,
         "Data Analyst", 0.03),
        (lambda s: s.get("devops", 0) > 0.8 and s.get("networking", 0) > 0.6,
         "DevOps Engineer", 0.04),
        (lambda s: s.get("javascript", 0) > 0.8 and s.get("design", 0) > 0.5,
         "Frontend Developer", 0.03),
    ]

    # Apply boosts
    for pred in ml_predictions:
        for condition, career, boost in rules:
            if condition(skill_vector) and pred["career"] == career:
                pred["confidence"] = min(pred["confidence"] + boost, 1.0)

    # Re-sort by confidence
    ml_predictions.sort(key=lambda x: -x["confidence"])

    # Re-assign ranks
    for i, pred in enumerate(ml_predictions):
        pred["rank"] = i + 1

    return ml_predictions
