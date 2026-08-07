"""Small decoder-only GPT package."""

from .config import ExperimentConfig
from .model import GPT

__all__ = ["ExperimentConfig", "GPT"]
__version__ = "0.1.0"

