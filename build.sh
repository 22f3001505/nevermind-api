#!/usr/bin/env bash
# Render build script for Never Mind backend
set -o errexit

pip install -r requirements.txt

# Generate training data + train model
python datasets/generate_data.py
python ml_models/train_model.py

# Django setup
python manage.py collectstatic --no-input
python manage.py migrate --no-input
