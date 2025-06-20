#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Start Flask app with Gunicorn
gunicorn -w 1 -b 0.0.0.0:8080 app:app --timeout 600 --log-level debug