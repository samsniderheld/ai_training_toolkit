"""
Utility functions for LLM-based image captioning
"""

import base64
import os
import openai

from PIL import Image

# Try to import Gemini SDK
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


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


def get_caption_openai(base64_image, system_prompt, model="gpt-4o"):
    """Generate caption using OpenAI GPT-4V"""
    response = openai.chat.completions.create(
        model=model,
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

    )
    return response.choices[0].message.content.strip()


def get_caption_gemini(image_path, system_prompt, model="gemini-2.0-flash-exp"):
    """Generate caption using Google Gemini"""
    if not GEMINI_AVAILABLE:
        raise ImportError("Google Generative AI package not installed. Install with: pip install google-genai")

    # The client gets the API key from the environment variable `GEMINI_API_KEY`
    # Also support GOOGLE_API_KEY as fallback
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    # Load the image from file
    image = Image.open(image_path)

    # Create the prompt with system prompt and user instruction
    full_prompt = f"{system_prompt}\n\nCreate an image caption for this picture"

    # Generate content using the new API structure
    response = client.models.generate_content(
        model=model,
        contents=[full_prompt, image]
    )

    return response.text.strip()


def get_caption(image_input, system_prompt, model="gpt-4o", provider="openai"):
    """
    Generate caption using specified LLM provider

    Args:
        image_input: For OpenAI: base64 encoded image string. For Gemini: image file path
        system_prompt: System prompt for the LLM
        model: Model name to use
        provider: LLM provider ('openai' or 'gemini')

    Returns:
        Generated caption string
    """
    if provider.lower() == "openai":
        return get_caption_openai(image_input, system_prompt, model)
    elif provider.lower() == "gemini":
        return get_caption_gemini(image_input, system_prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers: 'openai', 'gemini'")


def get_image_files(root_dir):
    """Recursively find all image files in a directory"""
    image_files = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_files.append(os.path.join(folder, file))
    return image_files


