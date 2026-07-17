"""
llm_client.py

Centralized OpenRouter client for the AI Contract Risk Analyzer.
All LLM interactions should go through this module.
"""

import logging
from openai import OpenAI
from config import Config

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Validate configuration
# -------------------------------------------------

Config.validate()

# -------------------------------------------------
# OpenRouter Client
# -------------------------------------------------

client = OpenAI(
    api_key=Config.OPENROUTER_API_KEY,
    base_url=Config.OPENROUTER_BASE_URL,
)

# -------------------------------------------------
# Chat Function
# -------------------------------------------------

def chat(
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1000,
):
    """
    Sends a prompt to OpenRouter and returns the generated text.
    """

    try:

        logger.info("Sending request to OpenRouter...")

        messages = []

        if system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        response = client.chat.completions.create(

            model=Config.OPENROUTER_MODEL,

            messages=messages,

            temperature=temperature,

            max_tokens=max_tokens,

            extra_headers={
                "HTTP-Referer": "https://github.com",
                "X-Title": "AI Contract Risk Analyzer"
            }

        )

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError(
                "OpenRouter returned an empty response."
            )

        answer = content.strip()

        logger.info("OpenRouter response received.")

        return answer

    except Exception as e:

        logger.exception("OpenRouter request failed.")

        raise RuntimeError(
            f"OpenRouter Error: {e}"
        )