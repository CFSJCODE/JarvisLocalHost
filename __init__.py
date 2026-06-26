"""Project marker for J.A.R.V.I.S. LocalHost.

The runtime modules live at the repository root because the FastAPI server
loads them directly from `app.py`. Keep this file side-effect free so importing
the project package never tries to load optional ML or OCR dependencies.
"""

__all__ = []
