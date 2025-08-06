"""
Abstract base class for image generators.
"""

from abc import ABC, abstractmethod
from PIL import Image

class BaseGenerator(ABC):
    """Abstract base class for image generation models."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipeline = None
        
    @abstractmethod
    def load_pipeline(self) -> None:
        """Load the generation pipeline."""
        pass
        
    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        **kwargs
    ) -> Image.Image:
        """Generate an image from a text prompt."""
        pass
          
    @abstractmethod
    def set_lora_scale(self, scale: float) -> None:
        """Set the scale for a specific LoRA adapter."""
        pass