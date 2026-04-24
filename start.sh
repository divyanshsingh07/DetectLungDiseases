#!/bin/bash
# Quick start script for the Respiratory Health AI application

echo "🚀 Starting Respiratory Health AI"
echo ""
echo "📁 Project layout:"
echo "  app.py                — Flask entry point"
echo "  src/                  — training + evaluation scripts"
echo "  models/               — trained model artifacts (.pth, .pkl, meta)"
echo "  datasets/             — raw image and tabular datasets"
echo "  templates/ static/    — Jinja templates and static assets"
echo "  tests/                — verification scripts"
echo "  docs/                 — project documentation"
echo "  evaluation_results/   — generated metrics, charts, JSON report"
echo ""
echo "📍 Access at: http://127.0.0.1:5000"
echo ""

cd "$(dirname "$0")"
python app.py
