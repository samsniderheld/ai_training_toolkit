"""
LLM-based Image Captioning Package

Provides utilities for generating image captions using OpenAI's GPT-4V model.
"""

from .llm_utils import (
    load_system_prompt,
    encode_image_to_base64,
    get_caption,
    get_image_files,
    caption_image_file,
    default_system_prompt
)

__version__ = "1.0.0"
__all__ = [
    "load_system_prompt",
    "encode_image_to_base64", 
    "get_caption",
    "get_image_files",
    "caption_image_file",
    "default_system_prompt"
]