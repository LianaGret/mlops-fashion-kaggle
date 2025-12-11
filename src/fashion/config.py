from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from fashion.console import print_warning

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    secret_env_file = PROJECT_ROOT / ".secret.env"

    env_exists = env_file.exists()
    secret_exists = secret_env_file.exists()

    if not env_exists and not secret_exists:
        print_warning("No .env or .secret.env files found")
        return

    if env_exists:
        load_dotenv(env_file)

    if secret_exists:
        load_dotenv(secret_env_file, override=True)
