"""Pytest configuration — adds cortana package to sys.path."""
import sys
from pathlib import Path

# Allow tests to import from the cortana package
sys.path.insert(0, str(Path(__file__).parent.parent))
