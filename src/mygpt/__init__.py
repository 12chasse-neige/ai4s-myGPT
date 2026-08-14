"""Small decoder-only GPT package."""

from .config import ExperimentConfig
from .model import GPT
from .tokenizer import BPETokenizer

__all__ = ["BPETokenizer", "ExperimentConfig", "GPT"]
__version__ = "0.1.0"
