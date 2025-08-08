"""
FLUX pipeline implementation using HuggingFace Diffusers.
"""

import torch
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
from typing import Optional, Union, List
from .base import BaseGenerator


class FluxGenerator(BaseGenerator):
    """FLUX image generator using HuggingFace Diffusers."""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-dev", controlnet_model_id: Optional[str] = None):
        super().__init__(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.lora_scale = 1.0
        self.controlnet_model_id = controlnet_model_id
        self.controlnet = None
        
    def load_pipeline(self) -> None:
        """Load the FLUX pipeline."""
        if self.controlnet_model_id:
            self.controlnet = FluxControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=self.torch_dtype
            )
            self.pipeline = FluxControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=self.controlnet,
                torch_dtype=self.torch_dtype
            )
        else:
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype
            )
        
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)
            
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        seed: Optional[int] = None,
        control_image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        controlnet_conditioning_scale: float = 1.0,
        control_mode: Optional[Union[int, List[int]]] = None,
        **kwargs
    ) -> Image.Image:
        """Generate an image from a text prompt."""
        if self.pipeline is None:
            self.load_pipeline()
            
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
            
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "generator": generator,
            **kwargs
        }
        
        if self.controlnet_model_id and control_image is not None:
            generation_kwargs["control_image"] = control_image
            generation_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            if control_mode is not None:
                generation_kwargs["control_mode"] = control_mode
            
        result = self.pipeline(**generation_kwargs)
        
        return result.images[0]
        
    def set_lora_scale(self, scale: float) -> None:
        """Set the scale for LoRA adapters."""
        self.lora_scale = scale
        if self.pipeline is not None and hasattr(self.pipeline, 'set_adapters'):
            try:
                self.pipeline.set_adapters(scale=scale)
            except Exception:
                pass
                
    def load_lora_weights(
        self, 
        pretrained_model_name_or_path_or_dict: str,
        adapter_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """Load LoRA weights into the pipeline."""
        if self.pipeline is None:
            self.load_pipeline()
            
        self.pipeline.load_lora_weights(
            pretrained_model_name_or_path_or_dict,
            adapter_name=adapter_name,
            **kwargs
        )
        
    def unload_lora_weights(self) -> None:
        """Unload LoRA weights from the pipeline."""
        if self.pipeline is not None:
            self.pipeline.unload_lora_weights()
            
    def enable_memory_efficient_attention(self) -> None:
        """Enable memory efficient attention to reduce VRAM usage."""
        if self.pipeline is not None:
            self.pipeline.enable_attention_slicing()
            
    def disable_memory_efficient_attention(self) -> None:
        """Disable memory efficient attention."""
        if self.pipeline is not None:
            self.pipeline.disable_attention_slicing()
            
    def load_controlnet(self, controlnet_model_id: str) -> None:
        """Load a ControlNet model and reinitialize the pipeline."""
        self.controlnet_model_id = controlnet_model_id
        self.pipeline = None
        self.load_pipeline()
        
    def unload_controlnet(self) -> None:
        """Remove ControlNet and revert to standard FLUX pipeline."""
        self.controlnet_model_id = None
        self.controlnet = None
        self.pipeline = None
        self.load_pipeline()


class FluxControlNetGenerators:
    """Factory class for common FLUX ControlNet configurations."""
    
    @staticmethod
    def canny(model_id: str = "black-forest-labs/FLUX.1-dev") -> FluxGenerator:
        """Create a FLUX generator with Canny ControlNet."""
        return FluxGenerator(
            model_id=model_id,
            controlnet_model_id="InstantX/FLUX.1-dev-controlnet-canny"
        )
    
    @staticmethod
    def depth(model_id: str = "black-forest-labs/FLUX.1-dev") -> FluxGenerator:
        """Create a FLUX generator with Depth ControlNet."""
        return FluxGenerator(
            model_id=model_id,
            controlnet_model_id="InstantX/FLUX.1-dev-controlnet-depth"
        )
    
    @staticmethod
    def union(model_id: str = "black-forest-labs/FLUX.1-dev") -> FluxGenerator:
        """Create a FLUX generator with Union ControlNet (multi-control)."""
        return FluxGenerator(
            model_id=model_id,
            controlnet_model_id="InstantX/FLUX.1-dev-controlnet-union"
        )
    
    @staticmethod
    def upscaler(model_id: str = "black-forest-labs/FLUX.1-dev") -> FluxGenerator:
        """Create a FLUX generator with Upscaler ControlNet."""
        return FluxGenerator(
            model_id=model_id,
            controlnet_model_id="jasperai/Flux.1-dev-Controlnet-Upscaler"
        )