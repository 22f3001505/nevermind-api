"""
Never Mind — Enhanced ML Training Pipeline
Trains RandomForest + GradientBoosting ensemble with hyperparameter tuning.
Target: 85%+ accuracy on 2100 samples across 15 IT careers.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


SKILL_COLUMNS = ["python", "javascript", "html_css", "sql", "problem_solving",
                 "ml_ai", "design", "networking", "devops", "communication"]


def train_model():
    # ── 1. Load Dataset ──
    dataset_path = os.path.join(os.path.dirname(__file__), "../datasets/careers.csv")
    df = pd.read_csv(dataset_path)

    print(f"{'='*60}")
    print(f"  NEVER MIND — ML Training Pipeline")
    print(f"{'='*60}")
    print(f"\n📊 Dataset: {len(df)} samples, {df['role'].nunique()} careers")
    print(f"   Samples per career: ~{len(df) // df['role'].nunique()}")
    print(f"   Feature dimensions: {len(SKILL_COLUMNS)}")
    print(f"   Careers: {sorted(df['role'].unique())}\n")

    # ── 2. Prepare Features ──
    X = df[SKILL_COLUMNS].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["role"])

    # ── 3. Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"🔀 Split: {len(X_train)} train / {len(X_test)} test\n")

    # ── 4. Build Ensemble ──
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

    # Train individual models first for comparison
    print("🔧 Training RandomForest (200 trees)...")
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   RF Accuracy: {rf_acc:.4f} ({rf_acc * 100:.1f}%)")

    print("🔧 Training GradientBoosting (150 trees)...")
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"   GB Accuracy: {gb_acc:.4f} ({gb_acc * 100:.1f}%)")

    # Ensemble via soft voting
    print("🔧 Training Ensemble (RF + GB soft vote)...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1.2, 1.0]  # slightly favor RF for stability
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred)
    print(f"   Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc * 100:.1f}%)\n")

    # ── 5. Select Best Model ──
    best_acc = max(rf_acc, gb_acc, ensemble_acc)
    if best_acc == ensemble_acc:
        best_model = ensemble
        best_name = "Ensemble (RF+GB)"
    elif best_acc == rf_acc:
        best_model = rf
        best_name = "RandomForest"
    else:
        best_model = gb
        best_name = "GradientBoosting"

    y_best = best_model.predict(X_test)
    print(f"✅ Best Model: {best_name} → {best_acc:.4f} ({best_acc * 100:.1f}%)\n")

    # ── 6. Cross-Validation ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
    print(f"📈 5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Fold scores: {[f'{s:.3f}' for s in cv_scores]}\n")

    # ── 7. Classification Report ──
    target_names = label_encoder.classes_
    print("📋 Classification Report:")
    print(classification_report(y_test, y_best, target_names=target_names))

    # ── 8. Feature Importance ──
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        # Ensemble: average feature importances from sub-models
        importances = np.mean([
            est.feature_importances_ for est in best_model.estimators_
        ], axis=0)

    print("🔍 Feature Importance:")
    for skill, imp in sorted(zip(SKILL_COLUMNS, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"   {skill:>18}: {imp:.4f} {bar}")
    print()

    # ── 9. Confusion Matrix Summary ──
    cm = confusion_matrix(y_test, y_best)
    print(f"📊 Confusion Matrix (diagonal = correct):")
    correct = cm.diagonal().sum()
    total = cm.sum()
    print(f"   Correct: {correct}/{total} ({correct/total*100:.1f}%)\n")

    # ── 10. Save Artifacts ──
    model_dir = os.path.dirname(__file__)

    model_path = os.path.join(model_dir, "career_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"💾 Model saved → {model_path}")

    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"💾 Label encoder saved → {encoder_path}")

    meta = {
        "skill_columns": SKILL_COLUMNS,
        "careers": list(target_names),
        "model_type": best_name,
        "accuracy": round(best_acc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "n_samples": len(df),
        "n_features": len(SKILL_COLUMNS),
        "feature_importances": dict(zip(SKILL_COLUMNS, [round(x, 4) for x in importances]))
    }
    meta_path = os.path.join(model_dir, "model_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"💾 Model metadata saved → {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Training Complete — {best_name} @ {best_acc*100:.1f}%")
    print(f"{'='*60}")

    return best_model, label_encoder


if __name__ == "__main__":
    train_model()
