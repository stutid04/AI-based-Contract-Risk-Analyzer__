"""
config.py

Central configuration module for the AI Contract Risk Analyzer.

Responsibilities:
- Load environment variables
- Validate required configuration
- Expose configuration values to the rest of the project
"""

import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()


class Config:
    """
    Application configuration.
    """

    # OpenRouter
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

    # Timeout (seconds)
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

    @classmethod
    def validate(cls):
        """
        Validate all required configuration values.
        """

        missing = []

        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")

        if not cls.OPENROUTER_MODEL:
            missing.append("OPENROUTER_MODEL")

        if not cls.OPENROUTER_BASE_URL:
            missing.append("OPENROUTER_BASE_URL")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
            )