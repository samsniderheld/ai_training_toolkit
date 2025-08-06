"""
Utility functions for LLM-based image captioning
"""

import base64
import os
import openai


def load_system_prompt(prompt_file_path=None):
    """Load system prompt from a text file or use default"""
    # If no file specified, use default
    default_prompt_path = os.path.join(os.path.dirname(__file__), "default_system_prompt.txt")

    if prompt_file_path is None:
        prompt_file_path = default_prompt_path

    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {prompt_file_path} not found. Using fallback default.")
        try:
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print("Default prompt file also not found. Using hardcoded fallback.")
            return default_system_prompt()
    except Exception as e:
        print(f"Error loading prompt file: {e}. Using fallback default.")
        try:
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return default_system_prompt()


def default_system_prompt():
    """Fallback system prompt if file loading fails"""
    return """You are a image captioning agent. Your job is to take an input command and an image and then respond with a series of captions and tags.
  The goal is to describe the image with as much detail as possible. The captions are designed to train a model on a mouse from logitech.

  Output Examples:

 LGTCH_MOUSE, close up, top view, neon lighting, plain environment.

 LGTCH_MOUSE, wide shot, profile view, flat lighting, creative offce, notebook, pens, monidtor, key board."""


def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_caption(base64_image, system_prompt):
    """Generate caption using OpenAI GPT-4V"""
    response = openai.chat.completions.create(
        model="gpt-4o",  # or "gpt-4-turbo", depending on your plan
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Create an image caption for this picture",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            },
            
        ],
        max_tokens=70,
    )
    return response.choices[0].message.content.strip()


def get_image_files(root_dir):
    """Recursively find all image files in a directory"""
    image_files = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_files.append(os.path.join(folder, file))
    return image_files


