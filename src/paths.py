"""Paths shared by every script, resolved from the repository root."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'dataset')
RESULTS = os.path.join(ROOT, 'results')
TABLES = os.path.join(RESULTS, 'tables')
FIGURES = os.path.join(RESULTS, 'figures')
