"""Module for pydantic AI agent models."""

import os

# Default values, can be overwritten by user
MODEL_NAME = None
PROVIDER = None
API_KEY = None


def validate_ai_backend():
    """
    Validate the AI backend configuration and set environment variables.

    This function ensures that a MODEL_NAME is set and that an API key is available
    either via the API_KEY variable or the corresponding environment variable.
    It also infers the PROVIDER from the MODEL_NAME if not explicitly set.
    """
    global PROVIDER

    if not MODEL_NAME:
        raise ValueError("ai_agents.MODEL_NAME is not set. Please provide a model name.")

    # Infer provider if not set
    if not PROVIDER:
        model_lower = MODEL_NAME.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            PROVIDER = "openai"
        elif "gemini" in model_lower:
            PROVIDER = "google"
        elif "claude" in model_lower:
            PROVIDER = "anthropic"
        elif "mistral" in model_lower:
            PROVIDER = "mistral"
        elif ":" in MODEL_NAME:
            # e.g. "google-gla:gemini..."
            provider_prefix = MODEL_NAME.split(":")[0]
            if "google" in provider_prefix:
                PROVIDER = "google"

    # Determine expected env var
    if PROVIDER in ["google", "google-gla", "google-vertex"]:
        env_var = "GOOGLE_API_KEY"
    elif PROVIDER == "anthropic":
        env_var = "ANTHROPIC_API_KEY"
    elif PROVIDER == "mistral":
        env_var = "MISTRAL_API_KEY"
    elif PROVIDER == "groq":
        env_var = "GROQ_API_KEY"
    elif PROVIDER == "azure":
        env_var = "AZURE_API_KEY"
    elif PROVIDER == "openai":
        env_var = "OPENAI_API_KEY"
    else:
        raise ValueError(
            f"Unknown provider '{PROVIDER}'. Cannot determine API key environment variable."
        )

    # Check API Key
    if API_KEY:
        os.environ[env_var] = API_KEY
    elif env_var not in os.environ:
        raise ValueError(
            f"No API key found for provider '{PROVIDER}'. Please set 'ai_agents.API_KEY' or environment variable '{env_var}'."
        )


from ._plan_agents import generate_beam_angles
from ._cst_agents import generate_voi_objectives

__all__ = [
    "generate_beam_angles",
    "generate_voi_objectives",
    "API_KEY",
    "MODEL_NAME",
    "PROVIDER",
    "validate_ai_backend",
]
