"""Pytest fixtures and global setup."""

import matplotlib

# Headless backend so tests don't try to open a Tk window (and don't fail on
# systems where tk isn't installed alongside Python).
matplotlib.use("Agg")
