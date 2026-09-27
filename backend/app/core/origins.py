"""CORS origins for deployed Nexus-Q frontends.

Set ``NEXUSQ_ALLOWED_ORIGINS`` on the backend host (comma-separated), e.g.:
    NEXUSQ_ALLOWED_ORIGINS=https://nexus-q.vercel.app,https://nexus-q-git-main.vercel.app
Local dev origins (5173/4173) are always allowed.
"""

from __future__ import annotations

import os

DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def allowed_origins() -> list[str]:
    """Parse the deployed-frontend origins from the environment."""
    extra = os.environ.get("NEXUSQ_ALLOWED_ORIGINS", "")
    origins = list(DEFAULT_ORIGINS)
    for candidate in (part.strip() for part in extra.split(",")):
        if candidate and candidate not in origins:
            origins.append(candidate)
    return origins
